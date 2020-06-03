#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:18:53 2020

@author: jackreid

https://www.tucson.ars.ag.gov/notebooks/uploading_data_2_gee.html

"""




def ConvertRadToRef(filenamelist):
    """CONVERT SET OF RADIANCE DATA GEOTIFFS TO SURFACE REFLECANCE DATA
        
    Args:
        filenamelist: str of filenames, each seperated by a return
           
    Returns:
        N/A
    
    Outputs:
        AnalyticMS_Reflect.tif for each filename input
    """
    
    #Import the PlanetUtilities functions
    import PlanetUtilities

    #Iterate through list and convert to reflectance
    for file in filenamelist.split():
        PlanetUtilities.Rad_To_Ref(file)
    
    
def UploadToGoogleCloudStorage(filenamelist, bucket):
    """UPLOAD SET OF FILES TO GOOGLE CLOUD STORAGE. CALLS THE UNIX COMMAND LINE INTERFACE
    
    Based on: https://www.tucson.ars.ag.gov/notebooks/uploading_data_2_gee.html
    
    Args:
        filenamelist: str of filenames to be uploaded, each seperated by a return
        bucket: name of bucket to upload files to
           
    Returns:
        N/A
    """
    
    #Import libraries
    import subprocess

    #Iterate through and upload each image
    for file in filenamelist.split():
        subprocess.call('gsutil -m cp ' + file + ' gs://' + bucket + '/',
                        shell=True)
        
    
def ImportIntoGEE(filenamelist, prefix, bucket, **kwargs):
    """IMPORTS A COLLECTION OF FILES IN GOOGLE CLOUD STORAGE INTO AN IMAGE COLLECTION 
    ON GOOGLE EARTH ENGINE
    
    Based on: https://www.tucson.ars.ag.gov/notebooks/uploading_data_2_gee.html
    
    Args:
        filenamelist: str of filenames to be imported, each seperated by a return
        prefix: prefix to be added to asset id names in GEE
        bucket: name of bucket to import the files from
        destination: optional; name of GEE image collection to add assets to; defaults to "New_Collection"
        suffix: optional; suffix to append to asset id names in GEE; defaults to nothing
        user: optional; username of GEE account, defaults to jackreid
        metadata: optional: binary flag for whether metadata is available for the files; defaults to 0
           
    Returns:
        N/A
        
    Outputs:
        Set of GEE assets in one image collection, each labeled in the form 
        prefix + [date of collection] + suffix
    """
    
    #Import libraries
    import subprocess
    from xml.dom import minidom
    import os
    import re
    
    #Load key word arguments
    if 'destination' in kwargs:
        destination = kwargs.pop('destination')
    else:
        destination = 'New_Collection'
    if 'suffix' in kwargs:
        suffix = kwargs.pop('suffix')
        suffix = '_' + suffix
    else:
        suffix = ''  
    if 'user' in kwargs:
        user_name = kwargs.pop('user')
    else:
        user_name = 'jackreid'
    if 'metadata' in kwargs:
        metadata_flag = kwargs.pop('metadata')
    else:
        metadata_flag = 0
        
    #Initiate null list to track asset id names to avoid overwrites
    asset_list = []
    
    #Iterate through each filename
    for file in filenamelist.split():
        
        #Identify base name of the file and the date of image to serve as central component of asset id        
        base_name = os.path.basename(os.path.normpath(file))
        folder_filepath = re.sub(base_name, '', file)
        folder_name = os.path.basename(os.path.normpath(folder_filepath))
        sep = '_'
        date_name = folder_name.split(sep,1)[0]
        
        #Concatante asset id name
        assetname = prefix + '_' + date_name
                
        #Check if asset id already exists, generate number to append if so
        assetflag = 0
        for i in asset_list: 
            if(i == assetname):
                assetflag += 1
        asset_list.append(assetname)
        if assetflag != 0:
            assetname = assetname + '_' + str(assetflag)
        
        #Acquire and add time stamp if metadata is available
        if metadata_flag == 1:
            
            #Identify filepath of the metadata file
            if not(file.endswith('AnalyticMS.tif')):
                sep = '_AnalyticMS'  
                snippet = file.split(sep,1)[1]
                metadata_filename = re.sub(snippet, '_metadata.xml', file)
            else:
                metadata_filename = re.sub('.tif', '_metadata.xml', file)
            
            #Pull time stamp
            xmldoc = minidom.parse(metadata_filename)
            nodes = xmldoc.getElementsByTagName("ps:acquisitionDateTime")
            rawdate = nodes[0].firstChild.data
            vdate = rawdate[0:-6]  
            timestring = ' --time_start=' + vdate
            
        #Otherwise, leave a blank timestamp
        else:
              timestring = ''
        
        #Generate command sequence for importing the asset
        assetstring = ' --asset_id=users/' + user_name + '/' + destination + '/' + assetname + suffix
        bucketstring = ' gs://' + bucket + '/' + base_name
        commandstring = 'earthengine upload image' + assetstring + timestring + bucketstring
        print(commandstring)
        
        #Run the command sequence
        subprocess.call(commandstring,
                        shell=True)
        
        
        
if str.__eq__(__name__, '__main__'): 
    
    import subprocess


    SR_filenames = subprocess.getoutput('find ./Images/LagoonTest/ -iname *AnalyticMS_SR.tif') 
    UDM_filenames = subprocess.getoutput('find ./Images/LagoonTest/ -iname *AnalyticMS_DN_udm.tif')
    MS_filenames = subprocess.getoutput('find ./Images/LagoonTest/ -iname *AnalyticMS.tif')
    RF_filenames = subprocess.getoutput('find ./Images/LagoonTest/ -iname *AnalyticMS_Reflect.tif')

    
    # ConvertRadToRef(MS_filenames)
    # UploadToGoogleCloudStorage(RF_filenames, 'lagoon_ms_528')
    ImportIntoGEE(RF_filenames, 'Lagoon', 'lagoon_ms_528',
                  destination = 'Lagoon_SR',
                  suffix = 'RF',
                  user = 'jackreid',
                  metadata = 1)
