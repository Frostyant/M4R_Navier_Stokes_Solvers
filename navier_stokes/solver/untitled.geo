//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, 0, 0, 1.0};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 4};
//+
Line(3) = {4, 3};
//+
Line(4) = {3, 2};
//+
Point(5) = {0.5, 0.4, 0, 1.0};
//+
Point(6) = {0.5, 0.5, 0, 1.0};
//+
Point(7) = {0.5, 0.6, 0, 1.0};
//+
Circle(5) = {5, 6, 7};
//+
Circle(6) = {7, 6, 5};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Curve Loop(2) = {5, 6};
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
Physical Curve("cylinder") = {5, 6};
//+
Physical Surface("surface") = {1};
