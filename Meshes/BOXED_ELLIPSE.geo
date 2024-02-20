// Gmsh project created on Mon May  1 09:08:41 2023

// Distances in [m]
wire_radius = 2.5e-4;

collector_start = 0.02;
collector_middle = 0.025;
collector_end = 0.03;
collector_height = 0.002;

inner_before_wire = -0.01;
inner_after_collector = 0.04;
inner_height = 0.01;

before_wire = -0.03;
after_collector = 0.07;
mesh_height = 0.02;

//Cell size
fine = 2.5e-5; 
coarse = 1.e-4;
very_coarse = 5.e-4;
extremely_coarse = 1.e-3; //Changed from 2s

// Wire emitter points
Point(3) = {0,0,0,fine};
Point(2) = {-wire_radius,0,0,fine};
Point(1) = {-2*wire_radius,0,0,fine};

Circle(1) = {3,2,1};

// Emitter Box
shift = 3*wire_radius;
Point(22) = {shift,0,0,fine};
Point(23) = {shift,shift+wire_radius,0,fine};
Point(24) = {-2*wire_radius-shift,shift+wire_radius,0,fine};
Point(25) = {-2*wire_radius-shift,0,0,fine};

Line(15) = {3,22};
Line(16) = {22,23};
Line(17) = {23,24};
Line(18) = {24,25};
Line(19) = {25,1};

// Elliptic collector points
Point(4) = {collector_start,0,0,coarse};
Point(5) = {collector_middle,collector_height,0,coarse};
Point(7) = {collector_middle,0,0,coarse};
Point(6) = {collector_end,0,0,coarse};

Line(2) = {22,4};
Ellipse(3) = {5,7,4,4};//{4,6,7,5};
Ellipse(14) = {5,7,6,6};


// Inner rectangle
Point(8) = {inner_before_wire,0,0,very_coarse};
Point(9) = {inner_before_wire,inner_height,0,very_coarse};
Point(10) = {inner_after_collector,inner_height,0,very_coarse};
Point(11) = {inner_after_collector,0,0,very_coarse};

Line(4) = {6,11};

Line(5) = {11,10};
Line(6) = {10,9};
Line(7) = {9,8};
Line(13) = {8,25};


// Mesh boundaries
Point(12) = {before_wire,0,0,extremely_coarse};
Point(13) = {before_wire,mesh_height,0,extremely_coarse};
Point(14) = {after_collector,mesh_height,0,extremely_coarse};
Point(15) = {after_collector,0,0,extremely_coarse};

Line(8) = {15,14};
Line(9) = {14,13};
Line(10) = {13,12};
Line(11) = {12,8};
Line(12) = {11,15};

//Define surface
Curve Loop(1) = {-18,-17,-16,2,-3,14,4:7,13}; // Inner Box
Curve Loop(2) = {8:11,-7,-6,-5,12}; // Outer loop
Curve Loop(3) = {-1,15:19}; // Emitter loop

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface(1) = {1,2,3};

// Set boundary ids
Physical Curve(1) = {1}; // Emitter
Physical Curve(2) = {3,14}; // Collector
Physical Curve(10) = {10}; // Inlet
Physical Curve(11) = {8}; // Outlet

// Meshing algorithms
Mesh.Algorithm = 2;
Mesh.RecombineAll = 3;





