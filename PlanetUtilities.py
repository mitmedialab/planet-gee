#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:46:48 2020

@author: jackreid
"""


def Rad_To_Ref(filename, **kwargs):
    """CONVERT PLANET LABS' PLANETSCOPE RADIANCE DATA (...AnalyticMS.tif) TO 
    SURFACE REFLECTANCE DATA. REQUIRES THE METADATA FILE (...AnalyticMS_metadata.xml)
    TO BE PLACED IN SAME FOLDER AS THE .tif. 
    
    Based on: https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/toar/toar_planetscope.ipynb
    
    Args:
        filename: filename of the AnalyticMS.tif to be convereted
        visualize: optional, if set to 1, will generate and save plot of NIR reflectance
           
    Returns:
        reflect_filename: filepath of the output reflectance geotiff (scaled)
    
    Outputs:
        AnalyticMS_Reflect.tif: geotiff of surface reflectance saved to same folder as the original tif
    """
    
    #Import relevant libraries
    import rasterio
    import numpy as np
    from xml.dom import minidom
    
    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(filename) as src:
        band_blue_radiance = src.read(1)
        
    with rasterio.open(filename) as src:
        band_green_radiance = src.read(2)
    
    with rasterio.open(filename) as src:
        band_red_radiance = src.read(3)
    
    with rasterio.open(filename) as src:
        band_nir_radiance = src.read(4)
        
    #Construct the metadata filename
    if filename.endswith('.tif'):
        xmlfilename = filename[:-4] + '_metadata.xml'
    
    #Load the metadate file
    xmldoc = minidom.parse(xmlfilename)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    
    # XML parser refers to bands by numbers 1-4, extracts conversion coefficients
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    print("Conversion coefficients: {}".format(coeffs))
        
    #Calculate reflectance values for each band
    band_blue_reflectance = band_blue_radiance * coeffs[1]
    band_green_reflectance = band_green_radiance * coeffs[2]
    band_red_reflectance = band_red_radiance * coeffs[3]
    band_nir_reflectance = band_nir_radiance * coeffs[4]
    print("Red band radiance is from {} to {}".format(np.amin(band_red_radiance), np.amax(band_red_radiance)))
    print("Red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
        

    # Here we include a fixed scaling factor. This is common practice and allows data to be saved as unsigned 16 bit integers
    print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance),
                                                                          np.amax(band_red_reflectance)))
    scale = 10000
    blue_ref_scaled = scale * band_blue_reflectance
    green_ref_scaled = scale * band_green_reflectance
    red_ref_scaled = scale * band_red_reflectance
    nir_ref_scaled = scale * band_nir_reflectance
    print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled),
                                                                        np.amax(red_ref_scaled)))
    
    # Set spatial characteristics of the output object to mirror the input
    scale_kwargs = src.meta
    scale_kwargs.update(
        dtype=rasterio.uint16,
        count = 4)
    
    # Write band calculations to a new raster file
    if filename.endswith('.tif'):
        reflect_filename = filename[:-4] + '_Reflect.tif'
    with rasterio.open(reflect_filename, 'w', **scale_kwargs) as dst:
            dst.write_band(1, blue_ref_scaled.astype(rasterio.uint16))
            dst.write_band(2, green_ref_scaled.astype(rasterio.uint16))
            dst.write_band(3, red_ref_scaled.astype(rasterio.uint16))
            dst.write_band(4, nir_ref_scaled.astype(rasterio.uint16))

    #If visualize flag is set to 1, generate plot of NIR reflectance
    if 'visualize' in kwargs:
        visualize = kwargs.pop('visualize')
    else:
        visualize = 0
        
    if visualize == 1:
        print('visualizing...')
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        
        """
        The reflectance values will range from 0 to 1. You want to use a diverging color scheme to visualize the data,
        and you want to center the colorbar at a defined midpoint. The class below allows you to normalize the colorbar.
        """
        class MidpointNormalize(colors.Normalize):
            """
            Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
            e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
            Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
            """
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)
        
            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
        
        
        # Set min/max values from reflectance range for image (excluding NAN)
        min=np.nanmin(band_nir_reflectance)
        max=np.nanmax(band_nir_reflectance)
        mid=0.20
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        
        # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        # note that appending '_r' to the color scheme name reverses it!
        cmap = plt.cm.get_cmap('RdGy_r')
        
        cax = ax.imshow(band_nir_reflectance, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
        
        ax.axis('off')
        ax.set_title('NIR Reflectance', fontsize=18, fontweight='bold')
        
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)
        
        if filename.endswith('.tif'):
            plot_filename = filename[:-4] + '_ReflectPlot.png'
        fig.savefig(plot_filename, dpi=200, bbox_inches='tight', pad_inches=0.7)
        
        plt.show()
        
    return reflect_filename



def UDM_Masking(filename, **kwargs):
    """GENERATE MASK BASED ON UDM FILE ASSOCIATED WITH A .tif 
    
    Based on: https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb
    
    Args:
        filename: filename of the ...udm.tif to be turned into a mask
        visualize: optional, if set to 1, will generate and save plot of NIR reflectance
        rgb: optional, if set to 1, will generate mask for only non-quality RGB bands, ignoring quality of NIR band
    
    Returns:
        udm_mask: binary mask from udm file
     """
    
    #Import relevant libraries
    from collections import OrderedDict
    import matplotlib.colors as colors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    # Utility functions for loading a UDM image and identifying 
    # binary representation as class labels
    def load_udm(udm_filename):
        '''Load single-band bit-encoded UDM as a 2D array.'''
        with rasterio.open(udm_filename, 'r') as src:
            udm = src.read()[0,...]
        return udm
    
    def get_udm_labels(udm):
        '''Get the interpretation of the UDM binary values'''    
        def get_label(v):
            if v == 0:
                label = 'clear'
            elif v == 1:
                label = 'blackfill'
            else:
                labels = []
                if v & int('10',2):
                    labels.append('cloud')
                if v & int('1111100',2):
                    bands = []
                    if v & int('100',2):
                        bands.append('Blue')
                    if v & int('1000', 2):
                        bands.append('Green')
                    if v & int('10000', 2):
                        bands.append('Red')
                    if v & int('100000', 2):
                        bands.append('Red-Edge')
                    if v & int('1000000', 2):
                        bands.append('NIR')
                    labels.append('missing/suspect {} data'.format(', '.join(bands)))
                    
                if not len(labels):
                    labels.append('{0:08b}'.format(v))
    
                label = ', '.join(labels)
                
            return label
    
        return OrderedDict((v, get_label(v)) for v in np.unique(udm))
    
    #Load UDM and labels
    udm = load_udm(filename)
    udm_labels = get_udm_labels(udm)
   
   #Functions for generating binary mask
    def udm_to_mask_all(udm_array):
        '''Create a mask from the udm, masking all pixels with quality concerns''' 
        return udm_array != 0
    
    def udm_to_mask_rgb(udm_array):
        '''Create a mask from the udm, masking only pixels with RGB quality concerns''' 
        # RGB quality concern Bits: 0, 1, 2, 3, 4, 
        test_bits =int('00011111',2) # check for bits 1,2,3,4
        bit_matches = udm_array & test_bits # bit-wise logical AND operator
        return bit_matches != 0 # mask any pixels that match test bits
    
    #Generate either all quality mask or rgb quality mask
    if 'rgb' in kwargs:
        rgb = kwargs.pop('rgb')
    else:
        rgb = 0
    if rgb == 1:
        udm_mask = udm_to_mask_rgb(udm)
    else:
        udm_mask = udm_to_mask_all(udm)

    if 'visualize' in kwargs:
        visualize = kwargs.pop('visualize')
    else:
        visualize = 0
        
    #If visualize flag is set to 1, generate plot of UDM mask
    if visualize == 1:
        print('visualizing...')

        #Function for generating mask plot
        def plot_classified_band(class_band, class_labels=None, cmap='rainbow',
                             title='Class Labels', figdim=10):
            fig = plt.figure(figsize=(figdim, figdim))
            ax = fig.add_subplot(1, 1, 1)
            imshow_class_band(ax, class_band, class_labels, cmap=cmap)
            ax.set_title(title)
            ax.set_axis_off()
            
        #Function for generating legend for plot
        def imshow_class_band(ax, class_band, class_labels=None, cmap='rainbow'):
            """Show classified band with colormap normalization and color legend. Alters ax in place.
            
            possible cmaps ref: https://matplotlib.org/examples/color/colormaps_reference.html
            """
            class_norm = _ClassNormalize(class_band)
            im = ax.imshow(class_band, cmap=cmap, norm=class_norm)
        
            try:
                # add class label legend
                # https://stackoverflow.com/questions/25482876
                # /how-to-add-legend-to-imshow-in-matplotlib
                color_mapping = class_norm.mapping
                colors = [im.cmap(color_mapping[k]) for k in class_labels.keys()]
                labels = class_labels.values()
        
                # https://matplotlib.org/users/legend_guide.html
                # tag: #creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
                patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
        
                ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
            except AttributeError:
                # class_labels not specified
                pass
        
        # https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
        class _ClassNormalize(colors.Normalize):
            """Matplotlib colormap normalizer for a classified band.
            
            Inspired by https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
            """
            def __init__(self, arry):
                # get unique unmasked values
                values = [v for v in np.unique(arry)
                          if not isinstance(v, np.ma.core.MaskedConstant)]
        
                # map unique values to points in the range 0-1
                color_ticks = np.array(range(len(values)), dtype=np.float) / (len(values) - 1)
                self._mapping = dict((v, ct) for v, ct in zip(values, color_ticks))
                
                # Initialize base Normalize instance
                vmin = 0
                vmax = 1
                clip = False
                colors.Normalize.__init__(self, vmin, vmax, clip)
            
            def __call__(self, arry, clip=None):
                '''Create classified representation of arry for display.'''
                # round array back to ints for logical comparison
                arry = np.around(arry)
                new_arry = arry.copy()
                for k, v in self._mapping.items():
                    new_arry[arry==k] = v
                return new_arry
            
            @property
            def mapping(self):
                '''property required for colors.Normalize classes
                
                We update the _mapping property in __init__ and __call__ and just
                return that property here.
                '''
                return self._mapping
        
        #Generate UDM Plot
        plot_classified_band(udm, class_labels=udm_labels, title='UDM')
        
        #Generate Mask Plot
        mask_class_labels = {0: 'unmasked', 1: 'masked'}
        mask_cmap = 'viridis' # looks better when just two colors are displayed
        if rgb == 1:
            plot_classified_band(udm_to_mask_rgb(udm),
                 class_labels=mask_class_labels,
                 cmap=mask_cmap,
                 title='Mask RGB Quality Issues')
        else:
            plot_classified_band(udm_mask,
                         class_labels=mask_class_labels,
                         cmap=mask_cmap,
                         title='Mask All Quality Issues')

    return udm_mask



def Visualize_Planet(filename, **kwargs):
    """VISUALIZE PLANET TIFF IMAGE
    
    Based on: https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/toar/toar_planetscope.ipynb
    
    Args:
        filename: filename of the AnalyticMS.tif to be convereted
        num_bands: optional, specifies the number of bands to use in the image, defaults to 1
        bands: optional, specifies the specific bands to be vizualized, defaults to NIR
        save: optional, filename suffix to save png of image to; if unspecified, no image is saved
    Returns:
        fig: matplotlib figure of visualization
    """
    
    #Import relevant libraries
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    
    #Load Keyword Arguments
    
    if 'num_bands' in kwargs:
        num_bands = kwargs.pop('num_bands')
    else:
        num_bands = 1 #defaults to a single band image
        
    if 'bands' in kwargs:
        input_bands = kwargs.pop('bands')
    else:
        input_bands = [4] #defaults to the NIR band
    if 'save' in kwargs:
        save_name = kwargs.pop('save')
    else:
        save_name = [] # defaults to not saving a png
    
    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(filename) as src:
        band_blue = src.read(1)
        
    with rasterio.open(filename) as src:
        band_green = src.read(2)
    
    with rasterio.open(filename) as src:
        band_red = src.read(3)
    
    with rasterio.open(filename) as src:
        band_nir = src.read(4)
    
    #Function to identifying bands
    def id_band(band):
        """
        Associate each band with common reference methods
        """
        if band in ['blue','Blue','b','B',1]:
            outband = band_blue
            print('blue')
        elif band in ['green','Green','g','G',2]:
            outband = band_green
            print('green')
        elif band in ['red','Red','r','R',3]:
            outband = band_red
            print('red')
        elif band in ['nir', 'NIR', 'infrared', 4]:
            outband = band_nir
            print('nir')
        else:
            outband = []
    
        return outband
    
    
    """
     For a single band image, you want to use a diverging color scheme to visualize the data,
     and you want to center the colorbar at a defined midpoint. The class below allows you to normalize the colorbar.
     """
    class MidpointNormalize(colors.Normalize):
        """
        Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
        e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
        Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
        """
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
    
    #Initate figure
    fig = plt.figure(figsize=(20,10))

    #For single band image
    if num_bands == 1:
        
        #select band
        band_select = input_bands[0]
        band_visualize = id_band(band_select)
    
        # Set min/max values from reflectance range for image (excluding NAN)
        min_band=np.nanmin(band_visualize)
        max_band=np.nanmax(band_visualize)
        mid_band=(max_band-min_band)/2
    
        ax = fig.add_subplot(111)
        
        # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
        # note that appending '_r' to the color scheme name reverses it!
        cmap = plt.cm.get_cmap('RdGy_r')
        
        #Generate image
        cax = ax.imshow(band_visualize, 
                        cmap=cmap, 
                        clim=(min_band, max_band), 
                        norm=MidpointNormalize(midpoint=mid_band,vmin=min_band, vmax=max_band))
        
        ax.axis('off') #supress axes
        ax.set_title(str(band_select), fontsize=18, fontweight='bold')
        
        #Add colorbar
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)
    
    #For three band image
    elif num_bands == 3:
        ax = fig.add_subplot(111)
        #Select appropriate bands
        band_visualize = [id_band(input_bands[0]), 
                          id_band(input_bands[1]),
                          id_band(input_bands[2])]
        
        # Normalize bands into 0.0 - 1.0 scale
        def normalize(array):
            array_min, array_max = array.min(), array.max()
            return (array - array_min) / (array_max - array_min)
        
        # Normalize each band
        norm1 = normalize(band_visualize[0])
        norm2 = normalize(band_visualize[1])
        norm3 = normalize(band_visualize[2])
        
        # Stack bands into one multidimensional array
        normstack = np.dstack((norm1, norm2, norm3))
        
        # View the color composite
        # cax = fig.imshow(normstack)
        cax = ax.imshow(normstack)
        
        ax.axis('off')

    #show plot    
    plt.show()

    #If so specified, save the image as a png    
    if save_name != []:
        if filename.endswith('.tif'):
            plot_filename = filename[:-4] + save_name + '.png'
        fig.savefig(plot_filename, dpi=200, bbox_inches='tight', pad_inches=0.7)
    
    return fig

    
if str.__eq__(__name__, '__main__'):
    # Ref_To_Rad('/home/jackreid/Google Drive/School/Research/Space Enabled/Code/Planet/Images/LagoonTest/2020_04_15/20200415_124525_0f34_3B_AnalyticMS.tif')    
    
    
    # udm_mask = UDM_Masking('./Images/JacarepaguáLagoas/2020_05_11/20200511_115801_0f3c/20200511_115801_0f3c_3B_AnalyticMS_DN_udm.tif',
    #             visualize=1)
    
    Visualize_Planet('./Images/pika/20200527_151256_1009_3B_AnalyticMS_SR.tif',
                      num_bands = 3,
                      bands = ['red', 'green', 'blue'])