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

fun mul_array X (n:real) = foldl (fn (x, s) => s@[x*n]) [] X;

fun mul X n =
  let
    val shape = shape X
  in
    rev (foldl (fn (L, s) => s@[(mul_array L n)]) [] X)
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

fun predict X W1 W2 = sigm (dot (sigm (dot X W1)) W2);

fun errorMSE IN OUT W1 W2 = mean (foldl (fn (X, s) => s@[(foldl (fn (x, ss) => ss+x*x) 0.0 X )] ) [] (sub (predict IN W1 W2) OUT));

fun backprop x y W1 W2 =
  let
    val predicted = predict x W1 W2
    val delta_error = hd (hd (sub predicted y))
    val a_learning_rate = 0.001
    (* update W2 *)
    val nW2 = add W2 (mul (transpose (sigm(dot x W1))) (delta_error*a_learning_rate)) (* added the transpose for working purpose, might be an error*)
    (* update W1 *)
    val nW1 = add W1 (mul (dot (transpose x) (transpose W2)) (delta_error * a_learning_rate)) (* tranpose was added here too, might be also wrong *)
  in
    (nW1, nW2)
  end;

fun merge_in_tuple X Y =
  #3 (iter (List.length(X)) (X, Y, []) (fn ( (x::xr), (y::yr), s ) => (xr, yr, ([x],[y])::s )));

fun backpropagation X Y W_hidden W_out =
  let
    val merged = merge_in_tuple X Y
    val interations = 3000
  in
    iter interations (W_hidden, W_out) (fn (W1_, W2_) => (foldl (fn ((x,y), (wi, wo)) => backprop x y wi wo ) (W1_, W2_) merged))
  end;

(* examples *)
val X = to_real [[0,0],[0,1],[1,0],[1,1]];
val Y = to_real [[1], [0], [0], [1]];
val W1 = rand_Matrix [2,2];
val W2 = rand_Matrix [2,1];

val (W1_, W2_) = backpropagation X Y W1 W2;
errorMSE X Y W1 W2;
errorMSE X Y W1_ W2_;
