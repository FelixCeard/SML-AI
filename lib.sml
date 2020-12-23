(*
  TO ADD:
    - shape check for each function, helps for debug etc..
    - custom int and real type for more flexible inputs
*)

(* Hyperparameter for the random generator *)
val nachkommastellen = ref 4;
val seed = ref 2534;
val b = ref 424;
val Xval = ref 1;
val interations = ref 1000;
val lr = ref 0.01;

exception matrix_not_compatible;
exception row_and_col_missmatch;

(* foldl and iter *)
fun foldl f s nil = s
| foldl f s (x::xr) = foldl f (f(x,s)) xr;
fun 'a iter (n:int) (s:'a) (f:'a ->'a) : 'a = if n<1 then s else iter (n-1) (f s) (f);

(* help functions *)
fun pow (x:real) y = if y = 0 then 1.0 else x * pow x (y-1);

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
    val m = Real.round(pow 10.0 (!nachkommastellen))
  in
    gen_random (!seed) X (!b) m
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

fun predict X W1 W2 B1 B2 =
  let
    val s = #1 (shape X)
  in
    sigm (add (dot (sigm (add (dot X W1) (create_batchlength B1 s))) W2) (create_batchlength B2 s))
  end;

fun errorMSE IN OUT W1 W2 B1 B2 = mean (foldl (fn (X, s) => s@[(foldl (fn (x, ss) => ss+x*x) 0.0 X )] ) [] (sub (predict IN W1 W2 B1 B2) OUT));

fun element X = hd (hd X);

fun merge_in_tuple X Y = #3 (iter (List.length(X)) (X, Y, []) (fn ( (x::xr), (y::yr), s ) => (xr, yr, s@[([x],[y])] )));

fun div_list_n_list (l1:real list) l2 = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[x/y]) ) (l2, []) l1);

fun div_m_n_m X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@[(div_list_n_list x y)] )) (Y, []) X);

fun backpropagation_new X Y W1 W2 B1 B2=
  let
    val batch_size = #1 (shape Y)
    val pr = predict X W1 W2 B1 B2
    val d = (sigm_deriv pr)
    val delta_W2 = mul (sub Y pr) (element d)
    val delta_W1 = mul (dot delta_W2 (transpose W2)) (element d)
    val dw1 = div_m (dot (transpose X) delta_W1) (Real.fromInt(batch_size))
    val a_sW1 = sigm (dot X W1)
    val dw2 = div_m (dot (transpose a_sW1) delta_W2) (Real.fromInt(batch_size))

    (* new weights *)
    val nw1 = add W1 (mul dw1 (!lr))
    val nw2 = add W2 (mul dw2 (!lr))
    (* new biases *)
    val nb1 = transpose (dot (transpose delta_W1) (const_Matrix [batch_size, 1] 1.0))
    val nb2 = transpose (dot (transpose delta_W2) (const_Matrix [batch_size, 1] 1.0))
    val nb1 = add B1 (mul nb1 (!lr))
    val nb2 = add B2 (mul nb2 (!lr))
  in
      ((nw1, nw2), (nb1,nb2))
  end;

fun backprop X Y W_hidden W_out B_hid B_out=
  let
    val merged = merge_in_tuple X Y
  in
    iter (!interations) ((W_hidden, W_out),(B_hid, B_out)) (fn ( ((wi, wo),(bi, bo)) ) => backpropagation_new X Y wi wo bi bo)
  end;
