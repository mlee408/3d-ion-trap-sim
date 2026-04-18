SetFactory("OpenCASCADE");

// =========================
// 1. IMPORT GEOMETRY
// =========================

Merge "rf.step";
Merge "dc.step";
Merge "ground.step";

// ---- Assign volume IDs (update if needed) ----
// Check once in GUI → Tools → Visibility → Volumes

rfVols()  = {1, 2};
dcVols()  = {3:38};
gndVols() = {39:43};

// =========================
// 2. TAG CONDUCTORS (BEFORE BOOLEAN)
// =========================

// These are for sanity check / debugging
Physical Surface(11) = Boundary{ Volume{rfVols()}; };   // RF pre
Physical Surface(12) = Boundary{ Volume{dcVols()}; };   // DC pre
Physical Surface(13) = Boundary{ Volume{gndVols()}; };  // GND pre

Physical Volume(21) = {rfVols()};
Physical Volume(22) = {dcVols()};
Physical Volume(23) = {gndVols()};

// =========================
// 3. CREATE VACUUM BOX
// =========================

Lx = 1.2;
Ly = 1.2;
Lz_bot = 0.3;
Lz_top = 0.8;

Box(1000) = {-Lx/2, -Ly/2, -Lz_bot, Lx, Ly, Lz_bot + Lz_top};

// Save OUTER boundary (important!)
outer() = Boundary{ Volume{1000}; };

// =========================
// 4. SUBTRACT ELECTRODES STEP-BY-STEP
// =========================

// ---- subtract RF ----
v1() = BooleanDifference{ Volume{1000}; Delete; }{ Volume{rfVols()}; Delete; };

b1() = Boundary{ Volume{v1()}; };
rfSurf() = b1();
rfSurf() -= outer();

// ---- subtract DC ----
v2() = BooleanDifference{ Volume{v1()}; Delete; }{ Volume{dcVols()}; Delete; };

b2() = Boundary{ Volume{v2()}; };
dcSurf() = b2();
dcSurf() -= outer();
dcSurf() -= rfSurf();

// ---- subtract GROUND ----
v3() = BooleanDifference{ Volume{v2()}; Delete; }{ Volume{gndVols()}; Delete; };

b3() = Boundary{ Volume{v3()}; };
gndSurf() = b3();
gndSurf() -= outer();
gndSurf() -= rfSurf();
gndSurf() -= dcSurf();

// =========================
// 5. FINAL PHYSICAL GROUPS (USED IN FEM)
// =========================

Physical Surface(1) = {rfSurf()};   // RF electrode
Physical Surface(2) = {gndSurf()};  // Ground
Physical Surface(3) = {dcSurf()};   // DC electrodes

Physical Surface(9) = {outer()};    // outer boundary (optional)

Physical Volume(100) = {v3()};      // vacuum

// =========================
// 6. MESH SETTINGS
// =========================

Mesh.CharacteristicLengthMin = 0.005;
Mesh.CharacteristicLengthMax = 0.05;

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh 3;
