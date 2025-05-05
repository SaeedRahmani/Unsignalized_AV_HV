# Lyft L5 Unsignalised Intersection

## Important Notes
* We can change the number of time frames in the future by changing the yaml file in the following location: `l5kit/examples/visualisation/visualisation_config.yaml`. I am not sure but it may increase the efficiency of working with data. 
* For M1 Macs, we should install numpy manually, and then we should also mannualy change np.float to np.float64 (because we should install newer versions of numpy that do not support np.float)
* The data in this project are stored differently from the recommended Lyft structure. For running the codes in l5kit_conflict, the data should be stored in this structure:


<pre> ``` 
  l5kit_conflict/
├── scenes-sample/
│   └── sample.zarr/
├── scenes-train_full/
│   ├── train.zarr/
│   └── train_full.zarr/
├── scenes-validate/
│   └── validate.zarr/ 
``` </pre>


## Extension on `l5kit`

* `class MapAPI`: modify its `get_bound()` method to parse the new MapElements of `Junction`, which is defined in the `road_network.protobuf` file. Now, the id, its lanes' id and its nodes' ids are retrieved in `MapAPI`. These information is useful since they will be used in the rasterization stage to determine whether a scene or a frame includes the unsignalised intersection `(Lat: 37.419144, Lon: -122.150620), id: sGK1` we are interested.
* `class SemanticRasterizer`: this class implements the rasterizer when using the `semantic_debug` map_type in the `visualization.yaml` file. It is useful when determine whether a scene or a frame includes the unsignalised intersection `(Lat: 37.419144, Lon: -122.150620), id: sGK1` we are interested.
* `class Dataset`: create a new Dataset class that can load the data and identify whether the intersection is included in the current scene/frame. It use the `SemanticRasterizer` as the rasterizer.
