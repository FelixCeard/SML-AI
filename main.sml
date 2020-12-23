(* import the matrix multiplcation lib *)
use "lib.sml";
(* use "ml.sml"; *)
(* dot only works with real *)

(* testing some basic functions *)
val a = to_real [[1,2,3],[4,5,6],[7,8,9]];
val b = to_real [[9,8,7],[6,5,4],[3,2,1]];
val Y = to_real [[1],[2],[3]]

val single = to_real [[1]];

val ra = [1.0,2.0,3.0,4.0,5.0,6.0];
val rb = [2.0,3.0,4.0,5.0,6.0,7.0];

(*  addition *)
(div_m_n_m a b);
(* examples *)
(* val x1 = [[1,2,3,4],[5,6,7,8]]; (* (2,4) *)
val x2 = [[1,2,3], [5,6,7], [9,10,11],[12,13,14]]; (* (4,3) *)

val a1 = [[1.0,2.0], [2.0,3.0]];
val a2 = [[2.0, 6.0], [3.0,2.0]];

dot (to_real x1) (to_real x2); *)
(* map *)
