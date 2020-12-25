use "../src/lib.sml";

interations := 2000;
lr := 0.000021;

(* create the data sets *)
val train_set_input = to_real [[0,0,1],[0,1,0],[0,0,0],[1,1,0]];
val train_set_output = to_real [[1],[1],[0],[1]];

val test_set_input = to_real [[1,0,1],[1,0,0],[0,1,1],[1,1,1]];
val test_set_output = to_real [[1],[1],[1],[0]];

val model = create_model [3,4,4,1]

(* train the model *)
val trained_model = backprop train_set_input train_set_output model;

(* compute the initial error *)
val error_epoch0 = MSE train_set_input train_set_output model;

(* compute the error from the first epcoch *)
val train_error_epoch1 = MSE train_set_input train_set_output trained_model;
val test_error_epoch1 = MSE test_set_input test_set_output trained_model;

(* print what the model is predicting *)
predict train_set_input trained_model;
train_set_output; (* print the true output *)

(* test prediction *)
predict test_set_input trained_model;
test_set_output; (* print the true output *)
