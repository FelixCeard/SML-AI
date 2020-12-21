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
  | rand_Matrix [x] = [iter x [] (fn s => s@[(Real.fromInt(Random.randRange (1,100000) (Random.rand(~1,1)))/100000.0)])]
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

fun create_batchlength X n = iter (n-1) X (fn (S) => (hd X)::S);

fun predict X W1 W2 B1 B2 =
  let
    val s = #1 (shape X)
  in
    sigm (add (dot (sigm (add (dot X W1) (create_batchlength B1 s))) W2) (create_batchlength B2 s))
  end;
fun errorMSE IN OUT W1 W2 B1 B2 = mean (foldl (fn (X, s) => s@[(foldl (fn (x, ss) => ss+x*x) 0.0 X )] ) [] (sub (predict IN W1 W2 B1 B2) OUT));

fun element X = hd (hd X);

fun merge_in_tuple X Y = #3 (iter (List.length(X)) (X, Y, []) (fn ( (x::xr), (y::yr), s ) => (xr, yr, ([x],[y])::s )));

fun backpropagation_new X Y W1 W2 B1 B2=
  let
    val lr = 0.1
    val batch_size = #1 (shape Y)
    val pr = predict X W1 W2 B1 B2
    val d = (sigm_deriv pr)
    val delta_W2 = mul (sub Y pr) (element d)
    val delta_W1 = mul (dot delta_W2 (transpose W2)) (element d)
    (* [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)] *)
    val dw1 = div_m (dot (transpose X) delta_W1) (Real.fromInt(batch_size))
    val a_sW1 = sigm (dot X W1)
    val dw2 = div_m (dot (transpose a_sW1) delta_W2) (Real.fromInt(batch_size))
    (* new weights *)
    val nw1 = sub W1 (mul dw1 lr)
    val nw2 = sub W2 (mul dw2 lr)
    (* new biases *)

    (* db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas] *)
    val nb1 = transpose (dot (transpose delta_W1) (const_Matrix [batch_size, 1] 1.0))
    val nb2 = transpose (dot (transpose delta_W2) (const_Matrix [batch_size, 1] 1.0))
  in
      ((nw1, nw2), (nb1,nb2))
  end;

(* ((sub W1 (const_Matrix [#1 (shape W1), #2 (shape W1)] (element error_W1) )), (sub W2 (const_Matrix [#1 (shape W2), #2 (shape W2)] (element error_W2) ))) *)

(* model i [3,30,1] *)
val X = to_real [[0,0,1],[0,1,1],[1,0,0],[1,1,0]];
val Y = to_real [[1], [0], [1], [0]];
val W1 = [[0.3502887 , 0.37240587, 0.33225671, 0.67396507, 0.43846589,
        0.95486267, 0.20902327, 0.89693301, 0.43037097, 0.8247782 ,
        0.88163443, 0.74563089, 0.00653311, 0.766579  , 0.63292008,
        0.24512432, 0.44254029, 0.05477903, 0.62786728, 0.38430413,
        0.49145834, 0.1745838 , 0.85134493, 0.95646285, 0.6545379 ,
        0.48592677, 0.93176954, 0.74304272, 0.27133111, 0.51496413],
       [0.49032223, 0.07352547, 0.83001752, 0.44211809, 0.56881998,
        0.22034804, 0.23200744, 0.11102183, 0.6901749 , 0.31182523,
        0.29272239, 0.3151257 , 0.69090063, 0.8266054 , 0.73522721,
        0.1968651 , 0.06339544, 0.34380875, 0.20797717, 0.49399243,
        0.14193177, 0.6498857 , 0.26502894, 0.66120901, 0.50312137,
        0.9570944 , 0.85354847, 0.82365243, 0.4004978 , 0.5347667 ],
       [0.76875298, 0.34996818, 0.3053279 , 0.97363342, 0.91997555,
        0.02247412, 0.94650701, 0.58390371, 0.84490942, 0.67313096,
        0.54450917, 0.77124074, 0.53868374, 0.22006508, 0.90985379,
        0.4015532 , 0.06815342, 0.59102263, 0.89635205, 0.61527965,
        0.57216095, 0.66107835, 0.41962655, 0.07676265, 0.51582903,
        0.74224085, 0.59834144, 0.03128014, 0.3598203 , 0.69190265]];
val W2 = [[0.31895511],
       [0.29746525],
       [0.71119901],
       [0.65969992],
       [0.03324036],
       [0.57140454],
       [0.2914784 ],
       [0.92594983],
       [0.96918626],
       [0.34031191],
       [0.66703574],
       [0.27343888],
       [0.8692205 ],
       [0.45741118],
       [0.58576384],
       [0.72413415],
       [0.09407108],
       [0.99588567],
       [0.93434184],
       [0.24709233],
       [0.69615713],
       [0.45366755],
       [0.10680006],
       [0.24313332],
       [0.2351746 ],
       [0.97425989],
       [0.82036935],
       [0.43396009],
       [0.31054767],
       [0.85550728]];
val B1 = [[0.13247796, 0.74831968, 0.37571318, 0.96783194, 0.71205416,
        0.36803244, 0.28283636, 0.92807027, 0.81039817, 0.94697112,
        0.39702808, 0.68329175, 0.61934854, 0.7669776 , 0.2179972 ,
        0.22868814, 0.68255822, 0.09036625, 0.10055106, 0.56480168,
        0.40816004, 0.55008315, 0.0210745 , 0.71376879, 0.5022142 ,
        0.31936697, 0.08961427, 0.49960856, 0.01225054, 0.60912678]];
val B2 = rand_Matrix [1,1];

val ((nw1, nw2), (nb1, nb2)) = backpropagation_new X Y W1 W2 B1 B2;

fun backprop_new X Y W_hidden W_out B_hid B_out=
  let
    val merged = merge_in_tuple X Y
    val interations = 5
  in
    iter interations ((W_hidden, W_out),(B_hid, B_out)) (fn ( ((wi, wo),(bi, bo)) ) => backpropagation_new X Y wi wo bi bo)
  end;

(* val (nw1, nw2) = backpropagation_new [hd X] [hd Y] W1 W2; *)
val ((nW1, nW2),(nb1, nb2)) = backprop_new X Y W1 W2 B1 B2;

val error_epoch0 = errorMSE X Y W1 W2 B1 B2;
val error_epoch1 = errorMSE X Y nW1 nW2 nb1 nb2;
val diff = error_epoch0 - error_epoch1;
predict X W1 W2 B1 B2;
predict X nW1 nW2 nb1 nb2;
Y;
(* if it's negative, the model did the oposite of learning *)
