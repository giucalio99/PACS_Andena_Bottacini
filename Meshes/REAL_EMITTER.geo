// Gmsh project created on Mon May  1 09:08:41 2023

// Distances in [m]
mesh_height = 0.015;
before_wire = -0.04;
electrode_distance = 0.02;
naca_length = 0.1;
after_collector = 2.*naca_length + electrode_distance;
wire_radius = 2.5e-5;

inner_before_wire = -0.01;
inner_after_collector = 3.*electrode_distance/2.+naca_length;
inner_height = 0.0075;

//Mesh size 
fine = 4.e-6; 
coarse = 1.5e-3;
very_coarse = 1.75e-3;
extremely_coarse = 2.e-3;

// Wire emitter points
Point(3) = {0,0,0,fine};
Point(2) = {-wire_radius,0,0,fine};
Point(1) = {-2*wire_radius,0,0,fine};

Circle(1) = {3,2,1};
Circle(2) = {1,2,3};

// Emitter Box
wshift = 3*wire_radius;
Point(22) = {wshift,-(wshift+wire_radius),0,fine};
Point(23) = {wshift,wshift+wire_radius,0,fine};
Point(24) = {-2*wire_radius-wshift,wshift+wire_radius,0,fine};
Point(25) = {-2*wire_radius-wshift,-(wshift+wire_radius),0,fine};

//Line(15) = {3,22};
Line(16) = {22,23};
Line(17) = {23,24};
Line(18) = {24,25};
Line(19) = {25,22};

// NACA0010 points
shift = electrode_distance;
Point(80) = {shift, 0, 0, coarse};

Point(90) = {shift + 0.0125 * naca_length,0.01578 * naca_length, 0, coarse};
Point(100) = {shift + 0.0250 * naca_length,0.02178 * naca_length, 0, coarse};
Point(110) = {shift + 0.0500 * naca_length,0.02962 * naca_length, 0, coarse};
Point(120) = {shift + 0.0750 * naca_length,0.03500 * naca_length, 0, coarse};
Point(130) = {shift + 0.1000 * naca_length,0.03902 * naca_length, 0, coarse};
Point(140) = {shift + 0.1500 * naca_length,0.04455 * naca_length, 0, coarse};
Point(150) = {shift + 0.2000 * naca_length,0.04782 * naca_length, 0, coarse};
Point(160) = {shift + 0.2500 * naca_length,0.04952 * naca_length, 0, coarse};
Point(170) = {shift + 0.3000 * naca_length,0.05002 * naca_length, 0, coarse};
Point(180) = {shift + 0.4000 * naca_length,0.04837 * naca_length, 0, coarse};
Point(190) = {shift + 0.5000 * naca_length,0.04412 * naca_length, 0, coarse};
Point(200) = {shift + 0.6000 * naca_length,0.03803 * naca_length, 0, coarse};
Point(210) = {shift + 0.7000 * naca_length,0.03053 * naca_length, 0, coarse};
Point(220) = {shift + 0.8000 * naca_length,0.02187 * naca_length, 0, coarse};
Point(230) = {shift + 0.9000 * naca_length,0.01207 * naca_length, 0, coarse};
Point(240) = {shift + 0.9500 * naca_length,0.00672 * naca_length, 0, coarse};

Point(250) = {shift + 1.0000 * naca_length, 0, 0, coarse};

Point(260) = {shift + 0.0125 * naca_length, -0.01578 * naca_length, 0, coarse};
Point(270) = {shift + 0.0250 * naca_length,-0.02178 * naca_length, 0, coarse};
Point(280) = {shift + 0.0500 * naca_length,-0.02962 * naca_length, 0, coarse};
Point(290) = {shift + 0.0750 * naca_length,-0.03500 * naca_length, 0, coarse};
Point(300) = {shift + 0.1000 * naca_length,-0.03902 * naca_length, 0, coarse};
Point(310) = {shift + 0.1500 * naca_length,-0.04455 * naca_length, 0, coarse};
Point(320) = {shift + 0.2000 * naca_length,-0.04782 * naca_length, 0, coarse};
Point(330) = {shift + 0.2500 * naca_length,-0.04952 * naca_length, 0, coarse};
Point(340) = {shift + 0.3000 * naca_length,-0.05002 * naca_length, 0, coarse};
Point(350) = {shift + 0.4000 * naca_length,-0.04837 * naca_length, 0, coarse};
Point(360) = {shift + 0.5000 * naca_length,-0.04412 * naca_length, 0, coarse};
Point(370) = {shift + 0.6000 * naca_length,-0.03803 * naca_length, 0, coarse};
Point(380) = {shift + 0.7000 * naca_length,-0.03053 * naca_length, 0, coarse};
Point(390) = {shift + 0.8000 * naca_length,-0.02187 * naca_length, 0, coarse};
Point(400) = {shift + 0.9000 * naca_length,-0.01207 * naca_length, 0, coarse};
Point(410) = {shift + 0.9500 * naca_length,-0.00672 * naca_length, 0, coarse};

Spline(70) = {80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250};
Spline(80) = {80,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,250};

// Inner rectangle
Point(8) = {inner_before_wire,-inner_height,0,very_coarse};
Point(9) = {inner_before_wire,inner_height,0,very_coarse};
Point(10) = {inner_after_collector,inner_height,0,very_coarse};
Point(11) = {inner_after_collector,-inner_height,0,very_coarse};

Line(4) = {8,11};
Line(5) = {11,10};
Line(6) = {10,9};
Line(7) = {9,8};


// Mesh boundaries
Point(12) = {before_wire,-mesh_height,0,extremely_coarse};
Point(13) = {before_wire,mesh_height,0,extremely_coarse};
Point(14) = {after_collector,mesh_height,0,extremely_coarse};
Point(15) = {after_collector,-mesh_height,0,extremely_coarse};

Line(8) = {15,14};
Line(9) = {14,13};
Line(10) = {13,12};
Line(11) = {12,15};
//Line(12) = {11,15};

//Define surface
Curve Loop(1) = {16:19, 70,-80, 4:7}; // Inner Box
Curve Loop(2) = {8:11,4:7}; // Outer loop
Curve Loop(3) = {1,2,16:19}; // Emitter loop

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface(1) = {1,2,3};

// Set boundary ids
Physical Curve(1) = {1,2}; // Emitter
Physical Curve(2) = {70}; // Collector up
Physical Curve(3) = {80}; // Collector down
Physical Curve(10) = {10}; // Inlet
Physical Curve(11) = {8}; // Outlet

// Meshing algorithms
Mesh.Algorithm = 2;
Mesh.RecombineAll = 3;



