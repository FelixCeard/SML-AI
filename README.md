# SML AI
This is a package for creating AIs in SML.
***
## How to download it
Clone this repository with ```git clone https://github.com/FelixCeard/SML-AI/``` or download [lib.sml](https://github.com/FelixCeard/SML-AI/blob/master/lib.sml).

## How to use it
Reference the [lib.sml](https://github.com/FelixCeard/SML-AI/blob/master/lib.sml) file in your SML script with
``` SML
use "lib.sml";
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

(* initialize the weights and biases (currently, only two layers are being used) *)
val Weight_hidden = rand_Matrix [2,3];
val Weight_out = rand_Matrix [3,1];
val Bias_hid = rand_Matrix [1,3];
val Bias_out = rand_Matrix [1,1];

(* To train the model, you currently have to use everything *)
val ((new_hidden_weights, new_output_weights), (new_hidden_bias, new_output_bias)) = backprop X Y Weights_hidden Weights_out Bias_hid Bias_out;

(* You can calculate the error with the function 'errorMSE' *)
val error = errorMSE X Y Weights_hidden Weights_out Bias_hidden
```
---
# Contact
To contact me, simply write me an emaill: felix@falkenbergs.de
