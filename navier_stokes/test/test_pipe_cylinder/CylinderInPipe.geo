//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {30, 0, 0, 1.0};
//+
Point(3) = {30, 20, 0, 1.0};
//+
Point(4) = {0, 20, 0, 1.0};
//+
Point(5) = {7.5, 10, 0, 1.0};
//+
Point(6) = {7.5, 11, 0, 1.0};
//+
Point(7) = {7.5, 9, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 6};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {6, 5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("inflow") = {1};
//+
Physical Curve("outflow") = {3};
//+
Physical Curve("bottom") = {2};
//+
Physical Curve("top") = {4};
//+
Physical Curve("cylinder") = {6, 5};
//+
Physical Surface("surface") = {1};
