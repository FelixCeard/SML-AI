use "../lib.sml";

interations := 500;
lr := 0.6;

(* create the data sets *)
val train_set_input = to_real [[0,0,1],[0,1,0],[0,0,0],[1,1,0]];
val train_set_output = to_real [[1],[1],[0],[1]];

val test_set_input = to_real [[1,0,1],[1,0,0],[0,1,1],[1,1,1]];
val test_set_output = to_real [[1],[1],[1],[0]];

(* initialize the weights *)
val hidden_size = 4
val W1 = rand_Matrix [3,hidden_size];
val W2 = rand_Matrix [hidden_size,1];
val B1 = rand_Matrix [1,hidden_size];
val B2 = rand_Matrix [1,1];

(* train the model *)
val ((nW1, nW2),(nb1, nb2)) = backprop train_set_input train_set_output W1 W2 B1 B2;

(* compute the initial error *)
val error_epoch0 = errorMSE train_set_input train_set_output W1 W2 B1 B2;

(* compute the error from the first epcoch *)
val train_error_epoch1 = errorMSE train_set_input train_set_output nW1 nW2 nb1 nb2;
val test_error_epoch1 = errorMSE test_set_input test_set_output nW1 nW2 nb1 nb2;

(* print what the model is predicting *)
predict train_set_input nW1 nW2 nb1 nb2;
train_set_output; (* print the true output *)

(* test prediction *)
predict test_set_input nW1 nW2 nb1 nb2;
test_set_output; (* print the true output *)
