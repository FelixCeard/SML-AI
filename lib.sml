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

fun row_to_int X Y =
  case List.length(Y) = List.length(X) of
      true => #2 (foldl (fn ((x:real), ((y::yr), n) ) => (yr, n+(x*y)) ) (X, 0.0) Y)
    | false => raise row_and_col_missmatch;

fun get_nth xs n = if n<0 orelse null xs then raise Subscript else if n=0 then hd xs else get_nth (tl xs) (n-1);

fun ncol [] = 0
  | ncol [x] = 1
  | ncol (x::xr) = 1 + ncol xr;

fun nth_col X n =
  rev (foldl (fn (L, s) => (get_nth L n)::s) [] X);

fun shape X = (List.length(X), List.length(hd X));

fun rand_Matrix nil = []
  | rand_Matrix [x] = [iter x [] (fn s => s@[(Real.fromInt(Random.randRange (1,100000) (Random.rand(1,1)))/100000.0)])]
  | rand_Matrix (x::xr) = iter x [] (fn x => (rand_Matrix xr)@x);

fun const_Matrix nil _ = []
  | const_Matrix [x] n = [iter x [] (fn s => s@[n])]
  | const_Matrix (x::xr) n = iter x [] (fn x => (const_Matrix xr n)@x);

fun tr nil = nil
  | tr (X:int list) = Real.fromInt(hd X)::tr (tl X);

(* list from int to real*)
fun to_real X =
  foldl (fn (L, s) => s@[(tr L)]) [] X;


fun mean (X) = (#2 (iter (List.length(X)) (X, 0.0) (fn ((x::xr), m) => (xr, (x+m))))) / Real.fromInt(List.length(X));
mean ([1.0,2.0,3.0]);

(* main functions *)
fun add X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@ [(add_list x y)] )) (X, []) Y);

fun sub X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@ [(sub_list x y)] )) (X, []) Y);

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

fun mul_array X (n:real) = foldl (fn (x, s) => (x*n)::s) [] X;

fun mul X n =
  let
    val shape = shape X
  in
    rev (foldl (fn (L, s) => s@(mul_array L n)) [] X)
end;


val X = to_real [[0,0],[0,1],[1,0],[1,1]];
val Y = to_real [[1], [0], [0], [1]];
val W1 = rand_Matrix [2,2];
val W2 = rand_Matrix [2,1];

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

fun predict X W1 W2 = foldl (fn (x, s) => s@[(map sigmoid x)]) [] (dot (foldl (fn (x, s) => s@[(map sigmoid x)]) [] (dot X W1)) W2);
predict X W1 W2;

fun back X Y W1_ W2_ =
  let
    val lr = 0.01
    val model_output = predict [X] W1_ W2_
    val output_deltas = const_Matrix [1,(#2 (shape W2_))] 0.0
    (* Output_layer error *)
    val output_error = sub model_output [Y]
    val delta_out = mult (sigm_deriv model_output) output_error
    val hidden_out = sigm (dot [X] W1_)
    val delta_hidden = mult (sigm_deriv hidden_out) (transpose (dot W2_ (transpose delta_out)))
    (* update *)
    val update_output = mul (outer (hd hidden_out) (hd delta_out)) lr
    val update_hidden = outer (X) (hd delta_hidden)
  in
   [((add W1_ update_hidden), (add W2_ update_output))]
  end;

fun newW Ws w1_ w2_ =
  let
    val l = List.length(Ws)
    val fl = 1.0/Real.fromInt(l)
  in
    foldl (fn ((dw1,dw2), (nw1, nw2)) => ((add nw1 (mul dw1 fl)), (add nw2 (mul dw2 fl)))) (w1_, w2_) Ws
  end;

fun train_fit_backpropagate X Y W1 W2 =
  newW (#1 (foldl (fn (x, (s, (y::yr))) => (s@back x y W1 W2, yr)) ([], Y) X)) W1 W2;
