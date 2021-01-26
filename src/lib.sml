(*
  TO ADD:
    - shape check for each function, helps for debug etc..
    - custom int and real type for more flexible inputs
*)

(* Hyperparameters *)
val decimal_places = ref 4;
val seed = ref 2534;
val generator_b = ref 424;
val Xval = ref 1;
val interations = ref 10;
val lr = ref 0.01;
val error_history = ref [~1.0];

exception matrix_not_compatible;
exception row_and_col_missmatch;

(* foldl and iter *)
fun foldl f s nil = s
| foldl f s (x::xr) = foldl f (f(x,s)) xr;

fun 'a iter (n:int) (s:'a) (f:'a ->'a) : 'a = if n<1 then s else iter (n-1) (f s) (f);

(* help functions *)
fun pow (x:real) y = if y = 0 then 1.0 else x*pow x (y-1);

fun add_list (l1:real list) l2 = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[x+y]) ) (l2, []) l1);

fun sub_list (l1:real list) l2 = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[x-y]) ) (l2, []) l1);

fun add X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@ [(add_list x y)] )) (Y, []) X);

fun sub X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@ [(sub_list x y)] )) (Y, []) X);

fun row_to_int X Y =
  case List.length(Y) = List.length(X) of
      true => #2 (foldl (fn ((x:real), ((y::yr), n) ) => (yr, n+(x*y)) ) (Y, 0.0) X)
    | false => raise row_and_col_missmatch;

fun get_nth xs n = if n<0 orelse null xs then raise Subscript else if n=0 then hd xs else get_nth (tl xs) (n-1);

fun ncol [] = 0
  | ncol [x] = 1
  | ncol (x::xr) = 1 + ncol xr;

fun nth_col X n =
  rev (foldl (fn (L, s) => (get_nth L n)::s) [] X);

fun shape X = (List.length(X), List.length(hd X));

fun gen_random a X b m = Real.fromInt((a*X + b) mod m)/Real.fromInt(m);

fun rand X =
  let
    val m = Real.round(pow 10.0 (!decimal_places))
  in
    gen_random (!seed) X (!generator_b) m
end;

fun update_X_for_rand _ =
  let
    val i = !Xval
  in
  Xval := i + 1
end;

fun get_rand_number X =
  let
    val b = update_X_for_rand X
in
    (rand (!Xval))
end;

fun rand_Matrix nil = []
  | rand_Matrix [x] = [iter x [] (fn s => s@[get_rand_number 1])]
  | rand_Matrix (x::xr) = iter x ([]) (fn x => (rand_Matrix xr)@x);

fun const_Matrix nil _ = []
  | const_Matrix [x] n = [iter x [] (fn s => s@[n])]
  | const_Matrix (x::xr) n = iter x [] (fn x => (const_Matrix xr n)@x);

fun tr nil = nil
  | tr (X:int list) = Real.fromInt(hd X)::tr (tl X);

fun to_real X = foldl (fn (L, s) => s@[(tr L)]) [] X;

fun mean (X) = (#2 (iter (List.length(X)) (X, 0.0) (fn ((x::xr), m) => (xr, (x+m))))) / Real.fromInt(List.length(X));

fun dot X Y =
  let
    val shape_X = shape X
    val shape_Y = shape Y
    val n_col = ncol Y
  in
    if (#2 shape_X) = (#1 shape_Y) then
      rev (foldl (fn (L, s) => (iter (#2 shape_Y) [] (fn ot => ot@[(row_to_int L (nth_col Y (List.length(ot))))])::s)) [] X)
    else
      raise matrix_not_compatible
end;

fun transpose nil = nil
  | transpose (X) =
    let
      val s = shape X
    in
      iter (#2 s) [] (fn s => s@[nth_col X (List.length(s))])
    end;

fun sigmoid x = (1.0/(1.0 + Math.exp(~x)));

fun sigmoid_deriv (x) = x * (1.0 - x);

fun mul_array X (n:real) = foldl (fn (x, s) => s@[x*n]) [] X;

fun mul X n =
  let
    val shape = shape X
  in
    (foldl (fn (L, s) => s@[(mul_array L n)]) [] X)
end;

fun mult_list (l1:real list) l2 = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[x*y]) ) (l2, []) l1);

fun mult X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[(mult_list x y)] )) (X, []) Y);

fun outer (X:real list) Y = foldl (fn (x, s) => s@[(foldl (fn (y, a) => a@[y*x]) [] Y )] ) [] X;

fun sigm X = foldl (fn (x, ot) => ot@[foldl (fn (x, s) => s@[sigmoid x]) [] x]) [] X;

fun sigm_deriv X = foldl (fn (x, ot) => ot@[foldl (fn (x, s) => s@[sigmoid_deriv x]) [] x]) [] X;

fun div_array L n = foldl (fn (x, s) => s@[(x/n)]) [] L;

fun div_m X n =
  let
    val shape = shape X
  in
     (foldl (fn (L, s) => s@[(div_array L n)]) [] X)
end;

fun create_batchlength X n = iter (n-1) X (fn (S) => (hd X)::S);

fun element X = hd (hd X);

fun merge_in_tuple X Y = #3 (iter (List.length(X)) (X, Y, []) (fn ( (x::xr), (y::yr), s ) => (xr, yr, [([x],[y])]@s )));

fun div_list_n_list (l1:real list) l2 = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[x/y]) ) (l2, []) l1);

fun div_m_n_m X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[(div_list_n_list x y)] )) (Y, []) X);

(* AI lib *)
fun create_layers weights biases =
  #2 (foldl (fn ( x , ( (b::br), lrs ) ) => (br, lrs@[(x, b)])) (biases,[]) weights);

fun predict X {weights, biases} =
  let
    val layers = create_layers weights biases
    val s = #1 (shape X)
  in
    (foldl (fn ((w, b) , out) => (sigm (add (dot out w) (create_batchlength b s)  ))) X layers)
  end;

fun create_model shapes =
  let
  val (ww, bb) = #2 (foldl (fn (current, (old, (w, b))) => (current,( w@[(rand_Matrix [old, current])], b@[(rand_Matrix [1, current])]))) ((hd shapes),([],[])) (tl shapes))
in
  {weights=ww, biases=bb}
end;

fun remove _ [] = []
  | remove n LIST = if n = 0 then LIST else remove (n-1) (tl LIST);

fun predict_nth_layer X n weights biases =
let
    val layers = create_layers weights biases
    val layers = rev (remove (List.length(weights) - n) (rev layers))
    val s = #1 (shape X)
  in
    (foldl (fn ((w, b) , out) => (sigm (add (dot out w) (create_batchlength b s)  ))) X layers)
  end;

fun backpropagation X Y {weights, biases}=
  let
    val batch_size = #1 (shape Y)
    val merged = merge_in_tuple weights biases

    (* last layer *)
    val pr = predict X {weights=weights, biases=biases}
    val d = (sigm_deriv pr)
    val delta_W2 = mul (sub Y pr) (element d)

    (* for all layers *)
    val deltas = #2 (foldl (fn (w,(ld, ds)) => ((mul (dot ld (transpose w)) (element d)), (((mul (dot ld (transpose w)) (element d))))::ds)) (delta_W2,[(delta_W2)]) ( (rev weights))) (* works *)
    val new_dw1 = div_m (dot (transpose X) (hd (tl deltas))) (Real.fromInt(batch_size)) (* works *)
    val dws = #3 (foldl (fn (dlt,(lst_dw, i, rst)) => (( div_m (dot (transpose (predict_nth_layer X i weights biases)) dlt) (Real.fromInt(batch_size)) ), i+1, rst@[( (* shape *) (( div_m (dot (transpose (predict_nth_layer X i weights biases)) dlt) (Real.fromInt(batch_size)) )))])) (new_dw1, 0, []) (tl deltas)) (* seems to work *)

    (* new weights *)
    val new_weights = #2 (foldl (fn (w, (dop::ds, s)) => (ds, s@[((*shape*) (add w (mul dop (!lr))))])) (dws, []) weights) (* seems to work *)

    (* new biases *)
    val nbs = foldl (fn (dlt,s) => s@[shape (transpose (dot (transpose dlt) (const_Matrix [batch_size, 1] 1.0)))]) [] (rev (tl (rev deltas))) (* weird that you remove sth. *)
    val new_biases = #2 (foldl (fn (bs, ((bias::bias_rest), rst)) => (bias_rest, rst@[(bias)])) (biases, []) nbs)
  in
      {weights = new_weights, biases = new_biases}
  end;

fun MSE X Y {weights, biases} =
  let
    val batch_size = Real.fromInt(#1 (shape X))
  in
    (foldl (fn (df, e) => (pow (foldl (fn (d,eo) => eo+(d*d)) 0.0 df) 2)+e ) 0.0 (sub (predict X {weights=weights, biases=biases}) Y)) / batch_size
  end;

fun make_history X Y {weights, biases} =
  let
    val error = MSE X Y {weights=weights, biases=biases}
    val history = (!error_history)
  in
    error_history := history@[error]
  end;

fun backprop X Y {weights, biases} =
  let
    val merged = merge_in_tuple X Y
  in
    iter (!interations) {weights=weights, biases=biases} (fn m =>
        let
          val added_error_to_history = make_history X Y m
        in
          backpropagation X Y m
        end)
  end;
