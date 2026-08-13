use building footprint database for generating the dataset
I would like you to integrate it into generating both positive and negative sample,but mantain the component based approach, so different approaches can be set in the config file

using earth engine (you can search the web how to use bfd within earth engine) get all buildings of above certain dimensions (minimal for a farm)

then if they are marked in osm or in farm transparency maps as a farm building use it as a positive example and if not then as a negative