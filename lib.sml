exception matrix_not_compatible;
exception row_and_col_missmatch;

(* foldl and iter *)
fun foldl f s nil = s
| foldl f s (x::xr) = foldl f (f(x,s)) xr;
fun 'a iter (n:int) (s:'a) (f:'a ->'a) : 'a = if n<1 then s else iter (n-1) (f s) (f);

(* help functions *)
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
  | rand_Matrix [x] = [iter x [] (fn s => s@[(Real.fromInt(Random.randRange (1,100000) (Random.rand()))/100000.0)])]
  | rand_Matrix (x::xr) = iter x [] (fn x => (rand_Matrix xr)@x);

  fun tr nil = nil
    | tr (X:int list) = Real.fromInt(hd X)::tr (tl X);

  (* list from int to real*)
  fun to_real X =
    foldl (fn (L, s) => s@[(tr L)]) [] X;

(* main functions *)
fun add X Y = #2 (foldl (fn (x, ((y::yr), ot)) => (yr, ot@ [(add_list x y)] )) (X, []) Y);

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
