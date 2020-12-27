# SML AI

This is a package for creating AIs in SML.
#
## How to use it
Reference the [lib.sml](https://github.com/FelixCeard/SML-AI/blob/master/src/lib.sml) file in your SML script with
``` SML
use "./path_to_file/lib.sml";
```
##### example code
``` SML
use "lib.sml";

(* hyperparameter can be adjusted like this *)
interations := 500;
lr := 0.6;

(* create the dataset *)
val X = to_real [[1,0],[0,1],[1,1],[0,0]]
val Y = to_real [[1],[1],[0],[0]]

(* initialize the weights and biases *)
val model = create_model [2,4,1];

(* To train the model, you currently have to use everything *)
val trained_model = backprop X Y model;

(* You can calculate the error with the function 'MSE' *)
val error = MSE X Y model
```
#
# WIki
You can find the wiki [here: https://github.com/FelixCeard/SML-AI/wiki](https://github.com/FelixCeard/SML-AI/wiki). Every function that is being used is explained there.

## How to download it
Clone this repository with ```git clone https://github.com/FelixCeard/SML-AI/``` or download [lib.sml](https://github.com/FelixCeard/SML-AI/blob/master/src/lib.sml).

#
# Contact
To contact me, simply write me an emaill: felix.ceard@gmail.com
