val X = to_real [[0,0],[0,1],[1,0],[1,1]];
val Y = to_real [[1], [0], [0], [1]];

val W1 = to_real [[1,1,1,1], [1,1,1,1]]; (* (2, 4) *)
val W2 = to_real [[1],[1],[1],[1]]; (* (4,1) *)
val W = to_real [[0],[1]];
val B = to_real [[1,1],[1,1],[1,1],[1,1]];


val model = (dot X W);

fun pow (x:real) y = if y = 0 then 1.0 else x * pow x (y-1);

fun mean X = (foldl (fn (x, s) => (hd x) + s) 0.0 X);

fun MSE X Y = mean(foldl (fn (L, (s::sr)) => sr@[[(pow ((hd s)-(hd L)) 2) / Real.fromInt(List.length(X))]]) Y model);

MSE model Y;

model; X; Y;
