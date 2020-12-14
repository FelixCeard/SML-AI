(* to be implemente : DIY EXP *)
fun get_decimal x = if Real.compare((Real.fromInt(Real.floor(x))), x) = EQUAL then Real.floor(x) else get_decimal (x*10.0);

fun n_digit z =
  case Real.compare((Real.fromInt(Real.floor(z))), z) of
    EQUAL => 0
  | _ => 1 + n_digit (z*10.0);

fun ggT(a, b) = if Real.compare(b, 0.0) = EQUAL then abs(a) else ggT(b, Real.fromInt((Real.floor(a)) mod (Real.floor(b))));

fun frac x n =
  let
    val b = (pow 10.0 (n))
    val a = (b*x)
    val gt = ggT(a,b)
  in
    ((a/gt),(b/gt))
end;

(* to be implemente : DIY EXP *)
