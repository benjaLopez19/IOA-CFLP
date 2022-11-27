option solver gurobi;
option display_round 5;
option solution_round 5;

param cli;
param loc;
#set facilities = {1 .. loc};
param facilities{1 .. loc};
param ICap{1 .. loc};
param FC{1 .. loc};
param dem{1 .. cli};
param TC{1 .. cli, 1 .. loc};
#var x {loc} binary;
#var x {facilities} >=0, <=1;
var y {1 .. cli, 1 .. loc} binary;
#var y {1 .. cli, facilities} >=0, <=1;

minimize Total_Cost: (sum {j in 1 .. loc} facilities[j] * FC[j]) + (sum {j in 1 .. loc} (sum {i in 1..cli} y[i,j] * TC[i,j])) ;

s.t.
allocation1 {i in 1..cli}:    	sum {j in 1 .. loc} y[i,j] = 1;
allocation2 {i in 1..cli, j in 1 .. loc}: y[i,j] <= facilities[j];
capacity {j in 1 .. loc}: sum {i in 1..cli} dem[i]*y[i,j] <= ICap[j]*facilities[j];
