//================================================================================================//
//------------------------------------------------------------------------------------------------//
//    Solid-Liquid phase flow calcuation with Moving Particle Hydrodynamics (Implicit)            //
//    Passively Moving Particle Solid is installed (Yokoyama Ryo)                                 //
//                                                                                                //
//================================================================================================//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <omp.h>

#ifdef _CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#endif

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
//#define TWO_DIMENSIONAL
//#define CONVERGENCE_CHECK

#define DIM 3
#define MELTING
#define OPENACC
#define emissivity 0.8
#define SBC 5.67e-8
#define Te  500
#define CUT  140

#define MAX_UNIQUE_PROPERTIES  1000


// Property definition
#define TYPE_COUNT   7
#define FLUID_BEGIN  0
#define FLUID_END    2
#define RIGID_BEGIN  2
#define RIGID_END    4
#define WALL_BEGIN   4
#define WALL_END     6

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod


#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *ParticleIndex;                // original particle id
static int *Property;                     // particle type
static int *RigidProperty;
static double (*Mass);                    // mass
static double (*Position)[DIM];           // coordinate
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total force acting on the particle
static int (*NeighborCount);                  // [ParticleCount]
static int (*Neighbor)[MAX_NEIGHBOR_COUNT];   // [ParticleCount]
static int (*NeighborCountP);                 // [ParticleCount]
static int (*NeighborP)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static int (*NeighborFluidCount);
static int (*NeighborRigidCount);
static double (*NeighborCalculatedPosition)[DIM];
static int    (*TmpIntScalar);                // [ParticleCount] to sort with cellId
static double (*TmpDoubleScalar);             // [ParticleCount]
static int    (*TmpIntRigid);
static double (*TmpDoubleVector)[DIM];        // [ParticleCount]
#define MARGIN (0.1*ParticleSpacing)
#pragma acc declare create(ParticleCount,ParticleIndex,Property,Mass,Position,Velocity,Force,NeighborCalculatedPosition)
#pragma acc declare create(NeighborCount,Neighbor,NeighborCountP,NeighborP,NeighborFluidCount,NeighborRigidCount)
#pragma acc declare create(TmpIntScalar,TmpDoubleScalar,TmpDoubleVector,TmpIntRigid)



// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth = 0.0;
static int CellCount[DIM];
static int TotalCellCount = 0;
static int *CellFluidParticleBegin;  // beginning of fluid particles in the cell
static int *CellFluidParticleEnd;  // end of fluid particles in the cell
static int *CellWallParticleBegin;   // beginning of wall particles in the cell
static int *CellWallParticleEnd;   // end of wall particles in the cell
static int *CellRigidParticleBegin;   // beginning of wall particles in the cell
static int *CellRigidParticleEnd;   // end of wall particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,TotalCellCount)
#pragma acc declare create(CellFluidParticleBegin,CellFluidParticleEnd,CellWallParticleBegin,CellWallParticleEnd,CellRigidParticleBegin,CellRigidParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
static double FrictionCoefficient[TYPE_COUNT];
static double SpringConstant[TYPE_COUNT];
static double AttenuationCoefficient[TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)
#pragma acc declare create(FrictionCoefficient,SpringConstant,AttenuationCoefficient)

// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double *VirialPressureInsideRadius; //VirialPressureInRegionInsideEffectiveRadius
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(FluidParticleBegin,FluidParticleEnd,DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP)
#pragma acc declare create(VirialPressureAtParticle,VirialPressureInsideRadius,VirialStressAtParticle,Mu,Lambda,Kappa)


static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallParticleBegin,WallParticleEnd,WallCenter,WallVelocity,WallOmega,WallRotation)

//RIGID particles //
static int *RigidCount;
static int (*RigidParticleBegin);
static int RigidBegin;
static int RigidEnd;
static int (*RigidParticleEnd);
static int RigidPropertyCount;
static int uniqueProperties[MAX_UNIQUE_PROPERTIES];
static double (*InertialTensorInverse)[DIM][DIM];
static double (*RigidPosition)[DIM];
static double (*RigidVelocity)[DIM];
static double (*AngularMomentum)[DIM]; // L=rwv;
static double (*AngularV)[DIM];
static double (*Quaternion)[4];
#pragma acc declare create(RigidCount,RigidParticleBegin,RigidBegin,RigidEnd,RigidParticleEnd)
#pragma acc declare create(RigidPropertyCount,uniqueProperties)
#pragma acc declare create(InertialTensorInverse,RigidPosition,RigidVelocity)
#pragma acc declare create(AngularMomentum,AngularV,Quaternion)

//Input Parameters for Melting and Solidification//
static double Heat[TYPE_COUNT];
static double ThermalConductivity[TYPE_COUNT];
static double MeltingPoint[TYPE_COUNT];
static double SpecificHeat[TYPE_COUNT];
static double SolidifyingEnthalpy[TYPE_COUNT];
static double LiquefyingEnthalpy[TYPE_COUNT];
static double CriticalSolidFraction[TYPE_COUNT];
#pragma acc declare create(Heat,ThermalConductivity,MeltingPoint,SpecificHeat,SolidifyingEnthalpy,LiquefyingEnthalpy,CriticalSolidFraction)


//Parameters for melting and solidifcation //
static double *Temperature;
static double *Enthalpy;
static double *SolidFraction;   //How amount of solid is contained in one particle
static double *Conductivity;
static double *MeltingTemp;
static double *Cp;
static double *H0;
static double *H1;
#pragma acc declare create(Temperature,Enthalpy,SolidFraction)
#pragma acc declare create(Conductivity,MeltingTemp,Cp,H0,H1)

//For the Structure concatnate
static double YoungModules[TYPE_COUNT];
static double *YoungModule;
//#pragma acc declare create(YoungModules,YoungModule)


// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );
static void initializeRigid( void );

static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static int neighborCalculation();
static void resetForce();
static void calculateCellParticle( void );
static void calculateNeighbor( void );
static void freeNeighbor( void );
static void calculatePhysicalCoefficients( void );
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateMatrixA( void );
static void freeMatrixA( void );
static void calculateMatrixC( void );
static void freeMatrixC( void );
static void multiplyMatrixC( void );
static void calculateMultiGridMatrix( void );
static void freeMultiGridMatrix( void );
static void solveWithConjugatedGradient( void );

//Rigid Calculation
static void calculateCenterofGravity();
static void calculateAngularMomentum();
static void calculateAngularV();
static void updateRigidNumber();
static void updateQuaternion();
static void calculateRigidBody();

//Spring-Dashpot Calculation
static void calculateTangentialDirectionForce();
static void calculateNormalDirectionForce();
static void calculatespring();

//For the melting and solidification calculation
static void calculateEnergyConservation();
static void calculateTemperature();
static void calculateSolidFraction();
static void calculateViscosity();
static void calculateRadiation();


static void calculateVirialPressureAtParticle();
static void calculateVirialPressureInsideRadius();
static void calculateVirialStressAtParticle();




// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}


#pragma acc routine seq
inline double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


    clock_t cFrom, cTill, cStart, cEnd;
    clock_t cNeigh=0, cExplicit=0, cImplicitTriplet=0, cImplicitInsert=0, cImplicitMatrix=0, cImplicitMulti=0, cImplicitSolve=0, cPrecondition=0, cVirial=0, cOther=0;

int main(int argc, char *argv[])
{
    
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
        int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
        if(argc>6)numberofthread=atoi(argv[6]);
    }
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
    {
        #ifdef _OPENMP
        omp_set_num_threads( numberofthread );
        #pragma omp parallel
        {
            if(omp_get_thread_num()==0){
                log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
            }
        }
        #endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeRigid();
    initializeDomain();
  


    #pragma acc update device(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
    #pragma acc update device(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],RigidProperty[0:ParticleCount],Mass[0:ParticleCount])
    #pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
    #pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
   #pragma acc update device(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
    #pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
   // #pragma acc update device(CellParticle[0:PowerParticleCount],CellIndex[0:PowerParticleCount])
    #pragma acc update device(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
    
    #ifdef MELTING
    
    #pragma acc update device(Temperature[0:ParticleCount],Enthalpy[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc update device(MeltingTemp[0:ParticleCount],Cp[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount])
     #pragma acc update device(ThermalConductivity[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],MeltingPoint[0:TYPE_COUNT],SolidifyingEnthalpy[0:TYPE_COUNT],LiquefyingEnthalpy[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
    #endif
         #pragma acc update device(SpringConstant[0:TYPE_COUNT],AttenuationCoefficient[0:TYPE_COUNT],FrictionCoefficient[0:TYPE_COUNT])
    
    
   // #pragma acc update device(RigidParticleBegin[0:RigidPropertyCount]) 
   // #pragma acc update device (RigidParticleEnd[0:RigidPropertyCount]) 
   // #pragma acc update device(InertialTensorInverse[0:RigidPropertyCount][0:DIM][0:DIM]) 
   // #pragma acc update device(RigidPosition[0:RigidPropertyCount][0:DIM]) 
   // #pragma acc update device(RigidVelocity[0:RigidPropertyCount][0:DIM]) 
  //  #pragma acc update device(AngularMomentum[0:RigidPropertyCount][0:DIM]) 
   // #pragma acc update device(AngularV[0:RigidPropertyCount][0:DIM])
   // #pragma acc update device(Quaternion[0:RigidPropertyCount][0:4])
    #pragma acc update device(Gravity[0:DIM])
    #pragma acc update device(WallParticleBegin,WallParticleEnd)
    #pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
    #pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)



    calculateCellParticle();
    calculateNeighbor();
    calculateDensityA();
    calculateGravityCenter();
    calculateDensityP();
    

    writeVtkFile("output.vtk");
   // freeNeighbor();
    
    {
        time_t t=time(NULL);
        log_printf("start main roop at %s\n",ctime(&t));
    }
    int iStep=(int)(Time/Dt);
    OutputNext = Time;
    VtkOutputNext = Time;
    cStart = clock();
    cFrom = cStart;
    while(Time < EndTime + 1.0e-5*Dt){
        
        if( Time + 1.0e-5*Dt >= OutputNext ){
            char filename[256];
            sprintf(filename,proffilename,iStep);
            writeProfFile(filename);
         //   log_printf("@ Prof Output Time : %e\n", Time );
            OutputNext += OutputInterval;
            cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
        }
              //changed
              //calculateConvection();

	// wall calculation
	calculateWall();
	// periodic boundary calculation
	calculatePeriodicBoundary();
	
	// reset Force to calculate conservative interaction
	resetForce();
	cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		
	// calculate Neighbor
	calculateCellParticle();
	calculateNeighbor();
	cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;
		
	// calculate density
	calculateDensityA();
	calculateGravityCenter();
	calculateDensityP();
	//calculateDivergenceP();
		
	// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
	calculatePhysicalCoefficients();
	
	// calculate P(s,rho) s:fixed
	calculatePressureA();
		
	// calculate diffuse interface force
	calculateDiffuseInterface();
		
	#ifdef MELTING
	// calculate Enrgy conservation equation
      	 calculateEnergyConservation();

         calculateRadiation();
     
       	//calculate solid fraction
      	 calculateSolidFraction();

   
        //calculate phase change
      	  calculateViscosity();
      	  #endif
      	  
      	           updateRigidNumber();		
        		
      	  	        //calculate normal direction
     	  calculateNormalDirectionForce();

        //calculate tangential direction
      	  calculateTangentialDirectionForce();
        

		
	// calculate Gravity
	calculateGravity();
		
	// calculate intermidiate Velocity
	calculateAcceleration();

	cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
 
       // calculate viscosity pressure implicitly
	calculateMatrixA();
	calculateMatrixC();
	cTill = clock(); cImplicitMatrix += (cTill-cFrom); cFrom = cTill;
	multiplyMatrixC();
	cTill = clock(); cImplicitMulti += (cTill-cFrom); cFrom = cTill;
	solveWithConjugatedGradient();
	
	// updateRigidNumber();
       
                       
        calculateCenterofGravity();
        calculateAngularMomentum();
        calculateAngularV();
        updateQuaternion();
        calculateRigidBody();
        //changed
        calculateConvection();
        
        cTill = clock(); cImplicitSolve += (cTill-cFrom); cFrom = cTill;
        
    
        
        if( Time + 1.0e-5*Dt >= VtkOutputNext ){
            calculateDivergenceP();
            calculatePressureP();
            calculateViscosityV();
            calculateVirialStressAtParticle();
            cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;
            char filename [256];
            sprintf(filename,vtkfilename,iStep);
            writeVtkFile(filename);
            log_printf("@ Vtk Output Time : %e\n", Time );
            VtkOutputNext += VtkOutputInterval;
            cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

        }
        
	//freeNeighbor();

        Time += Dt;
        iStep++;
        cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;

    }
    
    cEnd = cTill;
    
    {
        time_t t=time(NULL);
        log_printf("end main roop at %s\n",ctime(&t));
        log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
        log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
        log_printf("implicit triplet:        %lf [CPU sec]\n", (double)cImplicitTriplet/CLOCKS_PER_SEC);
        log_printf("implicit insert:         %lf [CPU sec]\n", (double)cImplicitInsert/CLOCKS_PER_SEC);
        log_printf("implicit matrix:         %lf [CPU sec]\n", (double)cImplicitMatrix/CLOCKS_PER_SEC);
        log_printf("implicit multiplication: %lf [CPU sec]\n", (double)cImplicitMulti/CLOCKS_PER_SEC);
        log_printf("precondition             %lf [CPU sec]\n", (double)cPrecondition/CLOCKS_PER_SEC);
        log_printf("implicit solve         : %lf [CPU sec]\n", (double)(cPrecondition+cImplicitSolve)/CLOCKS_PER_SEC);
        log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
        log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
        log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cImplicitTriplet+cImplicitInsert+cImplicitMatrix+cImplicitMulti+cPrecondition+cImplicitSolve+cVirial+cOther)/CLOCKS_PER_SEC);
        log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
    }

   
    return 0;

}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
            else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
            else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
            // else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
            else if(sscanf(buf," Density %lf %lf %lf %lf  %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3],&Density[4],&Density[5])==6){mode=reading_global;}
            else if(sscanf(buf," BulkModulus %lf %lf %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3],&BulkModulus[4],&BulkModulus[5])==6){mode=reading_global;}
            else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3],&BulkViscosity[4],&BulkViscosity[5])==6){mode=reading_global;}
            else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3],&ShearViscosity[4],&ShearViscosity[5])==6){mode=reading_global;}
            else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[2],&SurfaceTension[3],&SurfaceTension[4],&SurfaceTension[5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3],&InteractionRatio[0][4],&InteractionRatio[0][5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3],&InteractionRatio[1][4],&InteractionRatio[1][5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3],&InteractionRatio[2][4],&InteractionRatio[2][5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3],&InteractionRatio[3][4],&InteractionRatio[3][5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type4) %lf %lf %lf %lf %lf %lf",&InteractionRatio[4][0],&InteractionRatio[4][1],&InteractionRatio[4][2],&InteractionRatio[4][3],&InteractionRatio[4][4],&InteractionRatio[4][5])==6){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type5) %lf %lf %lf %lf %lf %lf",&InteractionRatio[5][0],&InteractionRatio[5][1],&InteractionRatio[5][2],&InteractionRatio[5][3],&InteractionRatio[5][4],&InteractionRatio[5][5])==6){mode=reading_global;}
            else if(sscanf(buf," YoungModules %lf %lf %lf %lf  ",&YoungModules[2],&YoungModules[3],&YoungModules[4],&YoungModules[5])==4){mode=reading_global;}
             else if(sscanf(buf,"SpringConstant %lf %lf %lf %lf  ",&SpringConstant[2],&SpringConstant[3],&SpringConstant[4],&SpringConstant[5])==4){mode=reading_global;}
            else if(sscanf(buf," FrictionCoefficient %lf %lf %lf %lf ",&FrictionCoefficient[2],&FrictionCoefficient[3],&FrictionCoefficient[4],&FrictionCoefficient[5])==4){mode=reading_global;}
            else if(sscanf(buf," AttenuationCoefficient %lf %lf %lf %lf ",&AttenuationCoefficient[2],&AttenuationCoefficient[3],&AttenuationCoefficient[4],&AttenuationCoefficient[5])==4){mode=reading_global;}
            else if(sscanf(buf," Enthalpy %lf %lf %lf %lf %lf %lf",&Heat[0],&Heat[1],&Heat[2],&Heat[3],&Heat[4],&Heat[5])==6){mode=reading_global;}
            else if(sscanf(buf," ThermalConductivity %lf %lf %lf %lf %lf %lf",&ThermalConductivity[0],&ThermalConductivity[1],&ThermalConductivity[2],&ThermalConductivity[3],&ThermalConductivity[4],&ThermalConductivity[5])==6){mode=reading_global;}
            else if(sscanf(buf," MeltingTemp %lf %lf %lf %lf %lf %lf",&MeltingPoint[0],&MeltingPoint[1],&MeltingPoint[2],&MeltingPoint[3],&MeltingPoint[4],&MeltingPoint[5])==6){mode=reading_global;}
            else if(sscanf(buf," SpecificHeat %lf %lf %lf %lf %lf %lf",&SpecificHeat[0],&SpecificHeat[1],&SpecificHeat[2],&SpecificHeat[3],&SpecificHeat[4],&SpecificHeat[5])==6){mode=reading_global;}
            else if(sscanf(buf," SolidifyingEnthalpy %lf %lf %lf %lf %lf %lf",&SolidifyingEnthalpy[0],&SolidifyingEnthalpy[1],&SolidifyingEnthalpy[2],&SolidifyingEnthalpy[3],&SolidifyingEnthalpy[4],&SolidifyingEnthalpy[5])==6){mode=reading_global;}
            else if(sscanf(buf," LiquefyingEnthalpy %lf %lf %lf %lf %lf %lf",&LiquefyingEnthalpy[0],&LiquefyingEnthalpy[1],&LiquefyingEnthalpy[2],&LiquefyingEnthalpy[3],&LiquefyingEnthalpy[4],&LiquefyingEnthalpy[5])==6){mode=reading_global;}
            else if(sscanf(buf," CriticalSolidFraction %lf %lf %lf %lf",&CriticalSolidFraction[0],&CriticalSolidFraction[1],&CriticalSolidFraction[2],&CriticalSolidFraction[3])==4){mode=reading_global;}
            else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
            else if(sscanf(buf," Wall4  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[4][0],  &WallCenter[4][1],  &WallCenter[4][2],  &WallVelocity[4][0],  &WallVelocity[4][1],  &WallVelocity[4][2],  &WallOmega[4][0],  &WallOmega[4][1],  &WallOmega[4][2])==9){mode=reading_global;}
            else if(sscanf(buf," Wall5  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[5][0],  &WallCenter[5][1],  &WallCenter[5][2],  &WallVelocity[5][0],  &WallVelocity[5][1],  &WallVelocity[5][2],  &WallOmega[5][0],  &WallOmega[5][1],  &WallOmega[5][2])==9){mode=reading_global;}
            else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
    #pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
#pragma acc enter data create(SpringConstant[0:TYPE_COUNT],FrictionCoefficient[0:TYPE_COUNT],AttenuationCoefficient[0:TYPE_COUNT])
#pragma acc enter data create(Heat[0:TYPE_COUNT],ThermalConductivity[0:TYPE_COUNT],MeltingPoint[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],LiquefyingEnthalpy[0:TYPE_COUNT],SolidifyingEnthalpy[0:TYPE_COUNT])
#pragma acc enter data create(Gravity[0:DIM],CriticalSolidFraction[0:TYPE_COUNT])
//#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd,RigidBegin,RigidEnd)
#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
    return;
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
    char buf[1024];

    try{
        if(fgets(buf,sizeof(buf),fp)==NULL)throw;
        sscanf(buf,"%lf",&Time);
        if(fgets(buf,sizeof(buf),fp)==NULL)throw;
        sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
               &ParticleCount,
               &ParticleSpacing,
               &DomainMin[0], &DomainMax[0],
               &DomainMin[1], &DomainMax[1],
               &DomainMin[2], &DomainMax[2]);
#ifdef TWO_DIMENSIONAL
        ParticleVolume = ParticleSpacing*ParticleSpacing;
#else
    ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
#endif
	ParticleIndex = (int *)malloc(ParticleCount * sizeof(int));
	Property = (int *)malloc(ParticleCount * sizeof(int));
	RigidProperty = (int *)malloc(ParticleCount * sizeof(int));
	Position = (double (*)[DIM])malloc(ParticleCount * sizeof(double[DIM]));
	
	Velocity = (double (*)[DIM])malloc(ParticleCount * sizeof(double[DIM]));
	DensityA = (double *)malloc(ParticleCount * sizeof(double));
	GravityCenter = (double (*)[DIM])malloc(ParticleCount * sizeof(double[DIM]));
	PressureA = (double *)malloc(ParticleCount * sizeof(double));
	VolStrainP = (double *)malloc(ParticleCount * sizeof(double));
	DivergenceP = (double *)malloc(ParticleCount * sizeof(double));
	PressureP = (double *)malloc(ParticleCount * sizeof(double));
	VirialPressureAtParticle = (double *)malloc(ParticleCount * sizeof(double));
	VirialPressureInsideRadius = (double *)malloc(ParticleCount * sizeof(double));
	VirialStressAtParticle = (double (*)[DIM][DIM])malloc(ParticleCount * sizeof(double[DIM][DIM]));
	Mass = (double *)malloc(ParticleCount * sizeof(double));
	Force = (double (*)[DIM])malloc(ParticleCount * sizeof(double[DIM]));
	Mu = (double *)malloc(ParticleCount * sizeof(double));
	Lambda = (double *)malloc(ParticleCount * sizeof(double));
	Kappa = (double *)malloc(ParticleCount * sizeof(double));
	YoungModule = (double *)malloc(ParticleCount * sizeof(double));

	// Parameters related to melting and solidification
	Temperature = (double *)malloc(ParticleCount * sizeof(double));
	Enthalpy = (double *)malloc(ParticleCount * sizeof(double));
	SolidFraction = (double *)malloc(ParticleCount * sizeof(double));
	Conductivity = (double *)malloc(ParticleCount * sizeof(double));
	MeltingTemp = (double *)malloc(ParticleCount * sizeof(double));
	Cp = (double *)malloc(ParticleCount * sizeof(double));
	H0 = (double *)malloc(ParticleCount * sizeof(double));
	H1 = (double *)malloc(ParticleCount * sizeof(double));

	#pragma acc enter data create(ParticleIndex[0:ParticleCount]) attach(ParticleIndex)
	#pragma acc enter data create(Property[0:ParticleCount]) attach(Property)
	#pragma acc enter data create(RigidProperty[0:ParticleCount]) attach(RigidProperty)
	#pragma acc enter data create(Position[0:ParticleCount][0:DIM]) attach(Position)
	#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM]) attach(Velocity)
	#pragma acc enter data create(DensityA[0:ParticleCount]) attach(DensityA)
	#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM]) attach(GravityCenter)
	#pragma acc enter data create(PressureA[0:ParticleCount]) attach(PressureA)
	#pragma acc enter data create(VolStrainP[0:ParticleCount]) attach(VolStrainP)
	#pragma acc enter data create(DivergenceP[0:ParticleCount]) attach(DivergenceP)
	#pragma acc enter data create(PressureP[0:ParticleCount]) attach(PressureP)
	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount]) attach(VirialPressureAtParticle)
	#pragma acc enter data create(VirialPressureInsideRadius[0:ParticleCount]) attach(VirialPressureInsideRadius)
	#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM]) attach(VirialStressAtParticle)
	#pragma acc enter data create(Mass[0:ParticleCount]) attach(Mass)
	#pragma acc enter data create(Force[0:ParticleCount][0:DIM]) attach(Force)
	#pragma acc enter data create(Mu[0:ParticleCount]) attach(Mu)
	#pragma acc enter data create(Lambda[0:ParticleCount]) attach(Lambda)
	#pragma acc enter data create(Kappa[0:ParticleCount]) attach(Kappa)
	//#pragma acc enter data create(YoungModule[0:ParticleCount]) attach(YoungModule)

	#pragma acc enter data create(Temperature[0:ParticleCount]) attach(Temperature)
	#pragma acc enter data create(Enthalpy[0:ParticleCount]) attach(Enthalpy)
	#pragma acc enter data create(SolidFraction[0:ParticleCount]) attach(SolidFraction)
	#pragma acc enter data create(Conductivity[0:ParticleCount]) attach(Conductivity)
	#pragma acc enter data create(MeltingTemp[0:ParticleCount]) attach(MeltingTemp)
	#pragma acc enter data create(Cp[0:ParticleCount]) attach(Cp)
	#pragma acc enter data create(H0[0:ParticleCount]) attach(H0)
	#pragma acc enter data create(H1[0:ParticleCount]) attach(H1)
	

		TmpIntScalar = (int *)malloc(ParticleCount*sizeof(int));
		TmpDoubleScalar = (double *)malloc(ParticleCount*sizeof(double));
		TmpDoubleVector = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
                Neighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
                NeighborCountP = (int *)malloc(ParticleCount*sizeof(int));
                NeighborP      = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
                NeighborFluidCount = (int *)malloc(ParticleCount*sizeof(int));
                NeighborRigidCount = (int *)malloc(ParticleCount*sizeof(int));
                NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

		#pragma acc enter data create(TmpIntScalar[0:ParticleCount]) attach(TmpIntScalar)
		#pragma acc enter data create(TmpDoubleScalar[0:ParticleCount]) attach(TmpDoubleScalar)
		#pragma acc enter data create(TmpDoubleVector[0:ParticleCount][0:DIM]) attach(TmpDoubleVector)
		#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
		#pragma acc enter data create(NeighborCountP[0:ParticleCount]) attach(NeighborCountP)
		#pragma acc enter data create(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(Neighbor)
		#pragma acc enter data create(NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(NeighborP)
		#pragma acc enter data create(NeighborFluidCount[0:ParticleCount]) attach(NeighborFluidCount)
		#pragma acc enter data create(NeighborRigidCount[0:ParticleCount]) attach(NeighborRigidCount)
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM]) attach(NeighborCalculatedPosition)
		
		// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
		ParticleCountPower=0;
		while((ParticleCount>>ParticleCountPower)!=0){
			++ParticleCountPower;
		}
		PowerParticleCount = (1<<ParticleCountPower);
		fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
		CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		#pragma acc enter data create(CellIndex   [0:PowerParticleCount]) attach(CellIndex)
		#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)

		
		#pragma acc update device(ParticleCountPower,PowerParticleCount)
		#pragma acc update device(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])


        double (*q)[DIM] = Position;
        double (*v)[DIM] = Velocity;

        for(int iP=0;iP<ParticleCount;++iP){
            if(fgets(buf,sizeof(buf),fp)==NULL)break;
            sscanf(buf,"%d   %d   %lf %lf %lf  %lf %lf %lf  %lf",
                   &Property[iP],
                   &RigidProperty[iP],
                   &q[iP][0],&q[iP][1],&q[iP][2],
                   &v[iP][0],&v[iP][1],&v[iP][2],
                   &Enthalpy[iP]
                   );
        }
    }catch(...){};

    fclose(fp);
    
    
    // set begin & end
    FluidParticleBegin=0;FluidParticleEnd=0;RigidBegin=0;RigidEnd=0;WallParticleBegin=0;WallParticleEnd=0;
    for(int iP=0;iP<ParticleCount;++iP){
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && RIGID_BEGIN<=Property[iP+1] && Property[iP+1]<RIGID_END){
            FluidParticleEnd=iP+1;
            RigidBegin=iP+1;
        }
        if(RIGID_BEGIN<=Property[iP] && Property[iP]<RIGID_END && WALL_BEGIN<=Property[iP+1] && Property[iP+1]<WALL_END){
            RigidEnd=iP+1;
            WallParticleBegin=iP+1;
        }
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && iP+1==ParticleCount){
            FluidParticleEnd=iP+1;
        }
        if(RIGID_BEGIN<=Property[iP] && Property[iP]<RIGID_END && iP+1==ParticleCount){
            RigidEnd=iP+1;
        }
        if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END && iP+1==ParticleCount){
            WallParticleEnd=iP+1;
        }
    }
    
      #pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount])
	#pragma acc update device(RigidProperty[0:ParticleCount])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd,RigidBegin,RigidEnd)
	
    return;
 
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

    const double (*q)[DIM] = Position;
   
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %d %e %e %e  %e %e %e\n",
                    Property[iP],
                    RigidProperty[iP],
                    q[iP][0], q[iP][1], q[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}


static void writeVtkFile(char *filename)
{
    
//    #pragma acc update host(ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
//    #pragma acc update host(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
//    #pragma acc update host(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
    #pragma acc update host(Temperature[0:ParticleCount],SolidFraction[0:ParticleCount],Enthalpy[0:ParticleCount])
//    #pragma acc update host(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
//    #pragma acc update host(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
//    #pragma acc update host(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount])
//    #pragma acc update host(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
    
    // update parameters to be output
    #pragma acc update host(Property[0:ParticleCount],RigidProperty[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
    #pragma acc update host(NeighborCount[0:ParticleCount],Mu[0:ParticleCount],NeighborFluidCount[0:ParticleCount],Force[0:ParticleCount][0:DIM],AngularV[0:RigidPropertyCount][0:DIM],AngularMomentum[0:RigidPropertyCount][0:DIM])
    
    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
    
    FILE *fp=fopen(filename, "w");
    
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");
    
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)q[iP][0], (float)q[iP][1], (float)q[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "\n");
    
    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "\n");
    

    fprintf(fp, "SCALARS Mass float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n",(float) Mass[iP]);
    }
    
        fprintf(fp, "SCALARS Rigid float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n",(float) RigidProperty[iP]);
    }
    
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS Viscosity float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Mu[iP]);
    }
        fprintf(fp, "\n");
    fprintf(fp, "SCALARS Temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Temperature[iP]);
    }
            fprintf(fp, "\n");
    fprintf(fp, "SCALARS Enthalpy float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Enthalpy[iP]);
    }
                fprintf(fp, "\n");
    fprintf(fp, "SCALARS SolidFraction float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)SolidFraction[iP]);
    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS PressureA float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//        fprintf(fp, "%e\n", (float)PressureA[iP]);
//    }
//    fprintf(fp, "\n");

    fprintf(fp, "\n");
    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for
    }

 
//    fprintf(fp, "SCALARS VirialPressureInsideRadius float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//        fprintf(fp, "%e\n", (float)VirialPressureInsideRadius[iP]); // trivial operation is done for
//    }
//    for(int iD=0;iD<DIM-1;++iD){
//        for(int jD=0;jD<DIM-1;++jD){
//            fprintf(fp, "\n");    fprintf(fp, "SCALARS VirialStressAtParticle[%d][%d] float 1\n",iD,jD);
//            fprintf(fp, "LOOKUP_TABLE default\n");
//            for(int iP=0;iP<ParticleCount;++iP){
//                fprintf(fp, "%e\n", (float)VirialStressAtParticle[iP][iD][jD]); // trivial operation is done for
//            }
//        }
//    }

      fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
    fprintf(fp, "\n");
    		fprintf(fp, "\n");
	fprintf(fp, "SCALARS particleId float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
		for(int iP=0;iP<ParticleCount;++iP){
				fprintf(fp, "%d\n", ParticleIndex[iP]);
			}
		fprintf(fp, "\n");
		fprintf(fp, "SCALARS cellId float 1\n");
		fprintf(fp, "LOOKUP_TABLE default\n");
		for(int iP=0;iP<ParticleCount;++iP){
				fprintf(fp, "%d\n", CellIndex[iP]);
			}
    
    //    fprintf(fp, "\n");
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");
//    fprintf(fp, "VECTORS GravityCenter float\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//        fprintf(fp, "%e %e %e\n", (float)GravityCenter[iP][0], (float)GravityCenter[iP][1], (float)GravityCenter[iP][2]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "VECTORS omega float\n");
    for(int iR=0;iR<ParticleCount;++iR){
    //   for(int iP=RigidParticleBegin[iR];iP<RigidParticleEnd[iR];iP++){
        fprintf(fp, "%e %e %e\n", (float)AngularV[iR][0], (float)AngularV[iR][1], (float)AngularV[iR][2]);
    }
 //   }
    fprintf(fp, "\n");
    
    fprintf(fp, "VECTORS momentum float\n");
    for(int iR=0;iR<RigidPropertyCount;++iR){
       for(int iP=RigidParticleBegin[iR];iP<RigidParticleEnd[iR];iP++){           
        fprintf(fp, "%e %e %e\n", (float)AngularMomentum[iR][0], (float)AngularMomentum[iR][1], (float)AngularMomentum[iR][2]);
    }
    }
    
    fprintf(fp, "\n");
    

    
    fflush(fp);
    fclose(fp);
}


static void initializeWeight()
{
    RadiusRatioG = RadiusRatioA;
    
    RadiusA = RadiusRatioA*ParticleSpacing;
    RadiusG = RadiusRatioG*ParticleSpacing;
    RadiusP = RadiusRatioP*ParticleSpacing;
    RadiusV = RadiusRatioV*ParticleSpacing;
    
    
    #ifdef TWO_DIMENSIONAL
    Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
    Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
    Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
    Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
    R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
    #else
    //code for three dimensional
    Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
    Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
    Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
    Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
    R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
    #endif
    
    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)ceil(radius_ratio);
        const int rangeX = range;
        const int rangeY = range;
        #ifdef TWO_DIMENSIONAL
        const int rangeZ = 0;
        #else
        const int rangeZ = range;
        #endif
        
        int count = 0;
        double sum = 0.0;
        for(int iX=-rangeX;iX<=rangeX;++iX){
            for(int iY=-rangeY;iY<=rangeY;++iY){
                for(int iZ=-rangeZ;iZ<=rangeZ;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                        const double x = ParticleSpacing * ((double)iX);
                        const double y = ParticleSpacing * ((double)iY);
                        const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
        
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }
    
    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)ceil(radius_ratio);
        const int rangeX = range;
        const int rangeY = range;
        #ifdef TWO_DIMENSIONAL
        const int rangeZ = 0;
        #else
        const int rangeZ = range;
        #endif

        int count = 0;
        double sum = 0.0;
        for(int iX=-rangeX;iX<=rangeX;++iX){
            for(int iY=-rangeY;iY<=rangeY;++iY){
                for(int iZ=-rangeZ;iZ<=rangeZ;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                        const double x = ParticleSpacing * ((double)iX);
                        const double y = ParticleSpacing * ((double)iY);
                        const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
    #pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
}


static void initializeFluid()
{
    for(int iP=0;iP<ParticleCount;++iP){
        Mass[iP]=Density[Property[iP]]*ParticleVolume;
    }
    for(int iP=0;iP<ParticleCount;++iP){
        Kappa[iP]=BulkModulus[Property[iP]];
    }
    for(int iP=0;iP<ParticleCount;++iP){
        Lambda[iP]=BulkViscosity[Property[iP]];
    }
    for(int iP=0;iP<ParticleCount;++iP){
        Mu[iP]=ShearViscosity[Property[iP]];
    }
      for(int iP=0;iP<ParticleCount;++iP){
        Enthalpy[iP]=Heat[Property[iP]];
    }

    #ifdef TWO_DIMENSIONAL
    CofK = 0.350778153;
    double integN=0.024679383;
    double integX=0.226126699;
    #else
    CofK = 0.326976006;
    double integN=0.021425779;
    double integX=0.233977488;
    #endif
    
    for(int iT=0;iT<TYPE_COUNT;++iT){
        CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
    }
    	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])
    
}

static void initializeWall()
{
    for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
        
        double theta;
        double normal[DIM]={0.0,0.0,0.0};
        double q[DIM+1];
        double t[DIM];
        double (&R)[DIM][DIM]=WallRotation[iProp];
        
        theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
        if(theta!=0.0){
            for(int iD=0;iD<DIM;++iD){
                normal[iD]=WallOmega[iProp][iD]/theta;
            }
        }
        q[0]=normal[0]*sin(theta*Dt/2.0);
        q[1]=normal[1]*sin(theta*Dt/2.0);
        q[2]=normal[2]*sin(theta*Dt/2.0);
        q[3]=cos(theta*Dt/2.0);
        t[0]=WallVelocity[iProp][0]*Dt;
        t[1]=WallVelocity[iProp][1]*Dt;
        t[2]=WallVelocity[iProp][2]*Dt;
        
        R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
        R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
        R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
        
        R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
        R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
        R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
        
        R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
        R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
        R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
    }
    
#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}



static void initializeRigid() {
    
        int uniquePropertyCount = 0;
        for (int iP = RigidBegin; iP < RigidEnd; ++iP) {
            int prop = RigidProperty[iP];
            int exists = 0;
            for (int iR = 0; iR < uniquePropertyCount; ++iR) {
                if (uniqueProperties[iR] == prop  && uniqueProperties[iR]!=-1  ) {
                    exists = 1;
                    break;
                }
            }
            if (!exists) {
                uniqueProperties[uniquePropertyCount++] = prop;
            }
        }
        RigidPropertyCount = uniquePropertyCount;
        
        log_printf("Unique Properties Count: %d %d\n",RigidPropertyCount,uniquePropertyCount);

        // 各ユニークなRigidPropertyに対応するRigidCountを計算
        RigidCount = (int *)malloc(RigidPropertyCount * sizeof(int));

        for (int iR = 0; iR < uniquePropertyCount; ++iR) {
            int count = 0;
            for (int iP = RigidBegin; iP < RigidEnd; ++iP) {
                if (RigidProperty[iP] == uniqueProperties[iR]) {
                    if(RigidProperty[iP] == 0) continue;
                    count++;
                }
            }
            RigidCount[iR] = count;
      //     log_printf("Rigid Count: %d\n", count);
        }

	TmpIntRigid = (int *)malloc(ParticleCount * sizeof(int));
        RigidParticleBegin = (int *)malloc(RigidPropertyCount * sizeof(int));
        RigidParticleEnd = (int *)malloc(RigidPropertyCount* sizeof(int));
        InertialTensorInverse = (double (*)[DIM][DIM])malloc(RigidPropertyCount * sizeof(double [DIM][DIM]));
        RigidPosition = (double (*)[DIM])malloc(RigidPropertyCount * sizeof(double [DIM]));
        RigidVelocity = (double (*)[DIM])malloc(RigidPropertyCount * sizeof(double [DIM]));
        AngularMomentum = (double (*)[DIM])malloc(RigidPropertyCount * sizeof(double [DIM]));
        AngularV = (double (*)[DIM])malloc(RigidPropertyCount * sizeof(double [DIM]));
        Quaternion = (double (*)[4])malloc(RigidPropertyCount * sizeof(double [4]));
      
        #pragma acc enter data create(RigidParticleBegin[0:RigidPropertyCount]) attach(RigidParticleBegin)
        #pragma acc enter data create(TmpIntRigid[0:ParticleCount]) attach(TmpIntRigid)
	#pragma acc enter data create(RigidParticleEnd[0:RigidPropertyCount]) attach(RigidParticleEnd)
	#pragma acc enter data create(InertialTensorInverse[0:RigidPropertyCount][0:DIM][0:DIM]) attach(InertialTensorInverse)
	#pragma acc enter data create(RigidPosition[0:RigidPropertyCount][0:DIM]) attach(RigidPosition)
	#pragma acc enter data create(RigidVelocity[0:RigidPropertyCount][0:DIM]) attach(RigidVelocity)
	#pragma acc enter data create(AngularMomentum[0:RigidPropertyCount][0:DIM]) attach(AngularMomentum)
	#pragma acc enter data create(AngularV[0:RigidPropertyCount][0:DIM]) attach(AngularV)
	#pragma acc enter data create(Quaternion[0:RigidPropertyCount][0:4]) attach(Quaternion)
	

        for (int iR = 0; iR < uniquePropertyCount; ++iR) {
            int prop = uniqueProperties[iR];
            int firstParticleIndex = -1;
            int lastParticleIndex = -1;
            for (int iP = RigidBegin; iP < RigidEnd; ++iP) {
           //     log_printf("No: %d\n",uniquePropertyCount );
           //     log_printf("Total Count: %d\n",RigidEnd-RigidBegin );
                if (RigidProperty[iP] == prop ) {
                    if (firstParticleIndex == -1) {
                        firstParticleIndex = iP;
                    }
                    lastParticleIndex = iP+1;
                }
            }
            RigidParticleBegin[iR] = firstParticleIndex;
            RigidParticleEnd[iR] = lastParticleIndex ; // +1 because end is exclusive
            // その他の初期化
            Quaternion[iR][0] = 1.0;
            Quaternion[iR][1] = 0.0;
            Quaternion[iR][2] = 0.0;
            Quaternion[iR][3] = 0.0;
            
            //ここでクォータニオンの初期値を設定する
            // Ryo Yokoyama 20231117
        }
        #pragma acc update device(TmpIntRigid[0:ParticleCount]) 
        #pragma acc update device(RigidParticleBegin[0:RigidPropertyCount]) 
	#pragma acc update device (RigidParticleEnd[0:RigidPropertyCount]) 
	#pragma acc update device(InertialTensorInverse[0:RigidPropertyCount][0:DIM][0:DIM]) 
	#pragma acc update device(RigidPosition[0:RigidPropertyCount][0:DIM]) 
	#pragma acc update device(RigidVelocity[0:RigidPropertyCount][0:DIM]) 
	#pragma acc update device(AngularMomentum[0:RigidPropertyCount][0:DIM]) 
	#pragma acc update device(AngularV[0:RigidPropertyCount][0:DIM])
	#pragma acc update device(Quaternion[0:RigidPropertyCount][0:4])

    }


static void initializeDomain( void )
{
	
  // Set the cell width to the particle spacing
    CellWidth = ParticleSpacing;

    // Initialize cell counts in each dimension
    double cellCount[DIM];
    cellCount[0] = round((DomainMax[0] - DomainMin[0]) / CellWidth);
    cellCount[1] = round((DomainMax[1] - DomainMin[1]) / CellWidth);
#ifdef TWO_DIMENSIONAL
    cellCount[2] = 1;  // For 2D case, set the cell count in Z direction to 1
#else
    cellCount[2] = round((DomainMax[2] - DomainMin[2]) / CellWidth);
#endif

    // Store cell counts as integers
    CellCount[0] = (int)cellCount[0];
    CellCount[1] = (int)cellCount[1];
    CellCount[2] = (int)cellCount[2];
    TotalCellCount = CellCount[0] * CellCount[1] * CellCount[2];
    fprintf(stderr, "TotalCellCount = %d\n", TotalCellCount);

    // Adjust domain boundaries if cell count is not an exact integer
    if (cellCount[0] != (double)CellCount[0] || cellCount[1] != (double)CellCount[1] || cellCount[2] != (double)CellCount[2]) {
        fprintf(stderr, "Warning: DomainWidth/CellWidth is not an integer value.\n");

        // Correct the domain maximum values to fit integer cell counts
        DomainMax[0] = DomainMin[0] + CellWidth * (double)CellCount[0];
        DomainMax[1] = DomainMin[1] + CellWidth * (double)CellCount[1];
        DomainMax[2] = DomainMin[2] + CellWidth * (double)CellCount[2];
        fprintf(stderr, "Adjusting the DomainMax values to: (%e, %e, %e)\n", DomainMax[0], DomainMax[1], DomainMax[2]);
    }

    // Calculate the width of the domain in each dimension
    DomainWidth[0] = DomainMax[0] - DomainMin[0];
    DomainWidth[1] = DomainMax[1] - DomainMin[1];
    DomainWidth[2] = DomainMax[2] - DomainMin[2];

    // Allocate memory for particle cell tracking arrays and check allocation success
    CellFluidParticleBegin = (int *)malloc(TotalCellCount * sizeof(int));
    CellFluidParticleEnd   = (int *)malloc(TotalCellCount * sizeof(int));
    CellWallParticleBegin  = (int *)malloc(TotalCellCount * sizeof(int));
    CellWallParticleEnd    = (int *)malloc(TotalCellCount * sizeof(int));
    CellRigidParticleBegin = (int *)malloc(TotalCellCount * sizeof(int));
    CellRigidParticleEnd   = (int *)malloc(TotalCellCount * sizeof(int));

    // Check if memory allocation was successful
    if (!CellFluidParticleBegin || !CellFluidParticleEnd || !CellWallParticleBegin || !CellWallParticleEnd ||
        !CellRigidParticleBegin || !CellRigidParticleEnd) {
        fprintf(stderr, "Error: Memory allocation failed for cell particle arrays.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate the minimum power for ParticleCount such that ParticleCount < 2^ParticleCountPower
    ParticleCountPower = 0;
    while ((ParticleCount >> ParticleCountPower) != 0) {
        ++ParticleCountPower;
    }
    PowerParticleCount = (1 << ParticleCountPower);  // Set the power of 2 for particles
    fprintf(stderr, "Memory allocated for CellIndex and CellParticle: %d elements\n", PowerParticleCount);

    // Allocate memory for cell index and cell particle arrays
    CellIndex    = (int *)malloc(PowerParticleCount * sizeof(int));
    CellParticle = (int *)malloc(PowerParticleCount * sizeof(int));

    // Check if memory allocation was successful
    if (!CellIndex || !CellParticle) {
        fprintf(stderr, "Error: Memory allocation failed for CellIndex or CellParticle arrays.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate the maximum radius among various particle properties
    MaxRadius = ((RadiusA > MaxRadius) ? RadiusA : MaxRadius);
    MaxRadius = ((2.0 * RadiusP > MaxRadius) ? 2.0 * RadiusP : MaxRadius);
    MaxRadius = ((RadiusV > MaxRadius) ? RadiusV : MaxRadius);
    fprintf(stderr, "MaxRadius = %lf\n", MaxRadius);
    
    #pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount]) attach(CellFluidParticleBegin)
	#pragma acc enter data create(CellFluidParticleEnd[0:TotalCellCount]) attach(CellFluidParticleEnd)
	#pragma acc enter data create(CellWallParticleBegin[0:TotalCellCount]) attach(CellWallParticleBegin)
	#pragma acc enter data create(CellWallParticleEnd[0:TotalCellCount]) attach(CellWallParticleEnd)
	#pragma acc enter data create(CellRigidParticleBegin[0:TotalCellCount]) attach(CellRigidParticleBegin)
	#pragma acc enter data create(CellRigidParticleEnd[0:TotalCellCount]) attach(CellRigidParticleEnd)


    // Update other related parameters to the GPU
    #pragma acc update device(MaxRadius, CellWidth, CellCount, TotalCellCount, DomainMax, DomainMin, DomainWidth)
    #pragma acc update device(ParticleCountPower,PowerParticleCount)

}

static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction (max:maxShift2)
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Mod(Position[iP][iD] - NeighborCalculatedPosition[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}


static void calculateCellParticle()
{
	// store and sort with cells
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellIndex[0:PowerParticleCount],Property[0:ParticleCount])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iP=0; iP<PowerParticleCount; ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			if(RIGID_BEGIN<=Property[iP] && Property[iP]<RIGID_END){
				CellIndex[iP] += TotalCellCount;
			}
			else if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
				CellIndex[iP] += 2*TotalCellCount;
			}
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = 3*TotalCellCount;
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	// sort with CellIndex
	// https://edom18.hateblo.jp/entry/2020/09/21/150416
	for(int iMain=0;iMain<ParticleCountPower;++iMain){
		for(int iSub=0;iSub<=iMain;++iSub){
			
			int dist = (1<< (iMain-iSub));
			
			#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
			#pragma acc loop independent
			#pragma omp parallel for
			for(int iP=0;iP<(1<<ParticleCountPower);++iP){
				bool up = ((iP >> iMain) & 2) == 0;
				
				if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
					int tmpCellIndex    = CellIndex[ iP ];
					int tmpCellParticle = CellParticle[ iP ];
					CellIndex[ iP ]     = CellIndex[ iP | dist ];
					CellParticle[ iP ]  = CellParticle[ iP | dist ];
					CellIndex[ iP | dist ]    = tmpCellIndex;
					CellParticle[ iP | dist ] = tmpCellParticle;
				}
			}
		}
	}
	
	// search for CellFluidParticleBegin[iC]
	#pragma acc kernels present (CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		CellFluidParticleBegin[iC]=0;
		CellFluidParticleEnd[iC]=0;
		CellWallParticleBegin[iC]=0;
		CellWallParticleEnd[iC]=0;
		CellRigidParticleBegin[iC]= 0;
		CellRigidParticleEnd[iC]=0;
	}
	
#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iP=1; iP<ParticleCount+1; ++iP){
		if( CellIndex[iP-1]<CellIndex[iP] ){
			if( CellIndex[iP-1] < TotalCellCount ){
				CellFluidParticleEnd[ CellIndex[iP-1] ] = iP;
			}
			else if( CellIndex[iP-1] - TotalCellCount < TotalCellCount ){
				CellRigidParticleEnd[ CellIndex[iP-1]-TotalCellCount ] = iP;
			}
			else if( CellIndex[iP-1] - 2*TotalCellCount < TotalCellCount ){
				CellWallParticleEnd[ CellIndex[iP-1]-2*TotalCellCount ] = iP;
			}
			if( CellIndex[iP] < TotalCellCount ){
				CellFluidParticleBegin[ CellIndex[iP] ] = iP;
			}
			else if( CellIndex[iP] - TotalCellCount < TotalCellCount ){
				CellRigidParticleBegin[ CellIndex[iP]-TotalCellCount ] = iP;
			}
			else if( CellIndex[iP] - 2*TotalCellCount < TotalCellCount ){
				CellWallParticleBegin[ CellIndex[iP]-2*TotalCellCount ] = iP;
			}
		}
	}
	
	// Fill zeros in CellParticleBegin and CellParticleEnd
	int power = 0;
	const int N = 3*TotalCellCount;
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	int * ptr = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(ptr[0:powerN])
	
	#pragma acc kernels present(ptr[0:powerN])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		ptr[iRow]=0;
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		ptr[iC]               =CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		ptr[iC+TotalCellCount]=CellRigidParticleEnd[iC] -CellRigidParticleBegin[iC];
		ptr[iC+2*TotalCellCount]=CellWallParticleEnd[iC] -CellWallParticleBegin[iC];
	}
	
	// Convert ptr to cumulative sum
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]+=ptr[iRow+dist];
		}
	}
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]-=ptr[iRow+dist];
			ptr[iRow+dist]+=ptr[iRow];
		}
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	//#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		if(iC==0){	CellFluidParticleBegin[iC]=0;	}
		else     { 	CellFluidParticleBegin[iC]=ptr[iC-1];	}
		CellFluidParticleEnd[iC]  =ptr[iC];
		CellRigidParticleBegin[iC] =ptr[iC-1+TotalCellCount];
		CellRigidParticleEnd[iC]   =ptr[iC+TotalCellCount];
		CellWallParticleBegin[iC] =ptr[iC-1+2*TotalCellCount];
		CellWallParticleEnd[iC]   =ptr[iC+2*TotalCellCount];
	}
	
	free(ptr);
	#pragma acc exit data delete(ptr[0:powerN])
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	{
		FluidParticleBegin = CellFluidParticleBegin[0];
		FluidParticleEnd   = CellFluidParticleEnd[TotalCellCount-1];
		RigidBegin = CellRigidParticleBegin[0];
		RigidEnd   = CellRigidParticleEnd[TotalCellCount-1];
		WallParticleBegin  = CellWallParticleBegin[0];
		WallParticleEnd    = CellWallParticleEnd[TotalCellCount-1];
	}
	#pragma acc update host(FluidParticleBegin,FluidParticleEnd,RigidBegin,RigidEnd,WallParticleBegin,WallParticleEnd)

/*
	// re-arange particles in CellIndex order
	#pragma acc kernels present(ParticleIndex[0:PowerParticleCount],TmpIntScalar[0:ParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=ParticleIndex[CellParticle[iP]];
	}
	#pragma acc kernels present(ParticleIndex[0:PowerParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],TmpIntScalar[0:ParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=Property[CellParticle[iP]];
	}
	#pragma acc kernels present(Property[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Property[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Position[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
		
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Velocity[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Velocity[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}

*/
}




static void calculateNeighbor( void )
{
#pragma acc kernels present (Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], NeighborFluidCount[0:ParticleCount],NeighborRigidCount[0:ParticleCount],NeighborCount[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborFluidCount[iP]=0;
        NeighborRigidCount[iP]=0;
        NeighborCount[iP]=0;
        for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
            Neighbor[iP][iN]=-1;
        }
    }
    
#pragma acc kernels present(NeighborCountP[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborCountP[iP]=0;
        for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
            NeighborP[iP][iN]=-1;
        }
    }
    
#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellRigidParticleBegin[0:TotalCellCount],CellRigidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],Position[0:ParticleCount][0:DIM])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int range = (int)(ceil((MaxRadius+MARGIN)/CellWidth));
        
   //     const int iCX=(CellIndex[iP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
   //     const int iCY=(CellIndex[iP]/CellCount[2])%CellCount[1];
   //     const int iCZ=CellIndex[iP]%CellCount[2];
        // same as
        const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
        const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
        const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];

#ifdef TWO_DIMENSIONAL
    #pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        const int jCZ =0;
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellFluidParticleBegin[jC];jCP<CellFluidParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                                NeighborCount[iP]++;
                                NeighborFluidCount[iP]++;
                                continue;
                            }
                            Neighbor[iP][NeighborCount[iP]]=jP;
                            NeighborCount[iP]++;
                            NeighborFluidCount[iP]++;
                            
                            if(qij2 <= RadiusP*RadiusP){
                                NeighborP[iP][NeighborCountP[iP]]=jP;
                                NeighborCountP[iP]++;
                            }
                        }
                    }
                }
            }
        
#pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
      const int jCZ =0;
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellRigidParticleBegin[jC];jCP<CellRigidParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                                NeighborCount[iP]++;
                                NeighborRigidCount[iP]++;
                                continue;
                            }
                            Neighbor[iP][NeighborCount[iP]]=jP;
                            NeighborCount[iP]++;
                            NeighborRigidCount[iP]++;
                            
                            if(qij2 <= RadiusP*RadiusP){
                                NeighborP[iP][NeighborCountP[iP]]=jP;
                                NeighborCountP[iP]++;
                            }
                        }
                    }
                }
            }

#pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        const int jCZ =0;
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellWallParticleBegin[jC];jCP<CellWallParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        NeighborCount[iP]++;
                        continue;
                    }
                    Neighbor[iP][NeighborCount[iP]]=jP;
                    NeighborCount[iP]++;
                    if(qij2 <= RadiusP*RadiusP){
                        NeighborP[iP][NeighborCountP[iP]]=jP;
                        NeighborCountP[iP]++;
                    }
                }
            }
        }
    }
    #else 
        
#pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        #pragma acc loop seq
        for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellFluidParticleBegin[jC];jCP<CellFluidParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                                NeighborCount[iP]++;
                                NeighborFluidCount[iP]++;
                                continue;
                            }
                            Neighbor[iP][NeighborCount[iP]]=jP;
                            NeighborCount[iP]++;
                            NeighborFluidCount[iP]++;
                            
                            if(qij2 <= RadiusP*RadiusP){
                                NeighborP[iP][NeighborCountP[iP]]=jP;
                                NeighborCountP[iP]++;
                            }
                        }
                    }
                }
            }
        }
        
#pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        #pragma acc loop seq
        for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellRigidParticleBegin[jC];jCP<CellRigidParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                                NeighborCount[iP]++;
                                NeighborRigidCount[iP]++;
                                continue;
                            }
                            Neighbor[iP][NeighborCount[iP]]=jP;
                            NeighborCount[iP]++;
                            NeighborRigidCount[iP]++;
                            
                            if(qij2 <= RadiusP*RadiusP){
                                NeighborP[iP][NeighborCountP[iP]]=jP;
                                NeighborCountP[iP]++;
                            }
                        }
                    }
                }
            }
        }
#pragma acc loop seq
for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
    #pragma acc loop seq
    for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        #pragma acc loop seq
        for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
            const int jC=CellId(jCX,jCY,jCZ);
            #pragma acc loop seq
            for(int jCP=CellWallParticleBegin[jC];jCP<CellWallParticleEnd[jC];++jCP){
                const int jP=CellParticle[jCP];
                double qij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                if(qij2 <= (MaxRadius)*(MaxRadius)){
                    if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        NeighborCount[iP]++;
                        continue;
                    }
                    Neighbor[iP][NeighborCount[iP]]=jP;
                    NeighborCount[iP]++;
                    if(qij2 <= RadiusP*RadiusP){
                        NeighborP[iP][NeighborCountP[iP]]=jP;
                        NeighborCountP[iP]++;
                    }
                }
            }
        }
    }
}
#endif
    }
        #pragma acc kernels present (Position[0:ParticleCount][0:DIM],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
        }
    }
    
}


static void calculateConvection()
{
	#pragma acc kernels present(Property[0:ParticleCount],SolidFraction[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(SolidFraction[iP]<CriticalSolidFraction[Property[iP]]){
        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
        }
    else{
        Position[iP][0] += Velocity[iP][0]*Dt/100;
        Position[iP][1] += Velocity[iP][1]*Dt/100;
        Position[iP][2] += Velocity[iP][2]*Dt/100;
    }
    }
    
	#pragma acc kernels present(RigidProperty[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
        for(int iP=RigidBegin;iP<RigidEnd;iP++){
        if(RigidProperty[iP] != -1){
            Position[iP][0] += Velocity[iP][0]*Dt;
            Position[iP][1] += Velocity[iP][1]*Dt;
            Position[iP][2] += Velocity[iP][2]*Dt;
        }
        else{
            Position[iP][0] += Velocity[iP][0]*Dt;
            Position[iP][1] += Velocity[iP][1]*Dt;
            Position[iP][2] += Velocity[iP][2]*Dt;
    }
    }
    }

static void resetForce()
{
    #pragma acc kernels present(Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{
    #pragma acc kernels present (Property[0:ParticleCount],Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mass[iP]=Density[Property[iP]]*ParticleVolume;
    }
    #pragma acc kernels present (Kappa[0:ParticleCount],Property[0:ParticleCount],VolStrainP[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
               Kappa[iP]=BulkModulus[Property[iP]];
        if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
    }
    
    #pragma acc kernels present(Lambda[0:ParticleCount],VolStrainP[0:ParticleCount],Property[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
  Lambda[iP]=BulkViscosity[Property[iP]];
     //   if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
    }
    
    #pragma acc kernels present (Property[0:ParticleCount],Mu[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mu[iP]=ShearViscosity[Property[iP]];
    }

    #pragma acc kernels present (Property[0:ParticleCount],Conductivity[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Conductivity[iP]=ThermalConductivity[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],MeltingTemp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        MeltingTemp[iP]=MeltingPoint[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],Cp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Cp[iP]=SpecificHeat[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],H0[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        H0[iP]=SolidifyingEnthalpy[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],H1[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        H1[iP]=LiquefyingEnthalpy[Property[iP]];
    }
    /*
        #pragma acc kernels present (Property[0:ParticleCount],YoungModule[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        YoungModule[iP]=YoungModules[Property[iP]];
    }
    */
    
}

static void calculateDensityA()
{
    
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
    {
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            double sum = 0.0;
          #pragma acc loop seq
	  #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
                if(iP==jP)continue;
                double ratio = InteractionRatio[Property[iP]][Property[jP]];
                double xij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double radius = RadiusA;
                const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
                if(radius*radius - rij2 >= 0){
                    const double rij = sqrt(rij2);
                    const double weight = ratio * wa(rij,radius);
                    sum += weight;
                }
            }
            DensityA[iP]=sum;
        }
    }
}

static void calculateGravityCenter()
{
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM])
    
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            double sum[DIM]={0.0,0.0,0.0};
          #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
                if(iP==jP)continue;
                double ratio = InteractionRatio[Property[iP]][Property[jP]];
                double xij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                }
                const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
                if(RadiusG*RadiusG - rij2 >= 0){
                    const double rij = sqrt(rij2);
                    const double weight = ratio * wg(rij,RadiusG);
                    #pragma acc loop seq
                    for(int iD=0;iD<DIM;++iD){
                        sum[iD] += xij[iD]*weight/R2g*RadiusG;
                    }
                }
            }
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                GravityCenter[iP][iD] = sum[iD];
            }
        }
    
}

static void calculatePressureA()
{

    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],PressureA[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
        if(N0a<=DensityA[iP]){
            PressureA[iP] = 0.0;
        }
    }
    
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureA[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double force[DIM]={0.0,0.0,0.0};
        #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
            double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
                const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD] += force[iD];
        }
    }
}

static void calculateDiffuseInterface()
{
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const double ai = CofA[Property[iP]]*(CofK)*(CofK);
        double force[DIM]={0.0,0.0,0.0};
     #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            const double aj = CofA[Property[iP]]*(CofK)*(CofK);
            double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
            double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(RadiusG*RadiusG - rij2 > 0){
                const double rij = sqrt(rij2);
                const double wij = ratio_ij * wg(rij,RadiusG);
                const double wji = ratio_ji * wg(rij,RadiusG);
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
                }
                const double dwij = ratio_ij * dwgdr(rij,RadiusG);
                const double dwji = ratio_ji * dwgdr(rij,RadiusG);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                double gr=0.0;
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]+=force[iD];
        }
    }
}

static void calculateDensityP()
{
    
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VolStrainP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double sum = 0.0;
        #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusP;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 >= 0){
                const double rij = sqrt(rij2);
                const double weight = wp(rij,radius);
                sum += weight;
            }
        }
        VolStrainP[iP] = (sum - N0p);
    }
}

static void calculateDivergenceP()
{

    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],DivergenceP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double sum = 0.0;
  #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusP;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 >= 0){
                const double rij = sqrt(rij2);
                const double dw = dwpdr(rij,radius);
                double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                double uij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    DivergenceP[iP] -= uij[iD]*eij[iD]*dw;
                }
            }
        }
        DivergenceP[iP]=sum;
    }
}

static void calculatePressureP()
{
    #pragma acc kernels present (PressureP[0:ParticleCount],Lambda[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],Kappa[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        PressureP[iP] = -Lambda[iP]*DivergenceP[iP];
        if(VolStrainP[iP]>0.0){
            PressureP[iP]+=Kappa[iP]*VolStrainP[iP];
        }
    }
    
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureP[0:ParticleCount],Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double force[DIM]={0.0,0.0,0.0};
        #pragma acc loop seq
	  #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusP;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dw = dwpdr(rij,radius);
                double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]+=force[iD];
        }
    }
}

static void calculateViscosityV(){

    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double force[DIM]={0.0,0.0,0.0};
       #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
            // viscosity term
            if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                double uij[DIM];
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
                }
                const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #ifdef TWO_DIMENSIONAL
                    force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
                    #else
                    force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
                    #endif
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD] += force[iD];
        }
    }
}


static void calculateNormalDirectionForce(){
    
#pragma acc kernels present (Property[0:ParticleCount],RigidProperty[0:ParticleCount],Position[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
#pragma acc loop independent
#pragma omp parallel for
    for( int iP=0;iP<ParticleCount;iP++){
    if(Property[iP]>=FLUID_BEGIN && Property[iP]<FLUID_END) continue;
        double force[DIM] ={0.0,0.0,0.0};
        #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            double vij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                vij[iD] = Velocity[jP][iD]-Velocity[iP][iD];
            }
            double v_n[DIM];
              if(RigidProperty[iP] != RigidProperty[jP] && RigidProperty[iP]!= -1 &&  RigidProperty[jP]!= -1){
            if(rij2<=ParticleSpacing*ParticleSpacing){
                 const double rij = sqrt(rij2);
                 double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;iD++){
                    v_n[iD] = vij[iD]*eij[iD];
                }
                 const double DampingConstant = -2*log(AttenuationCoefficient[Property[iP]])*sqrt(Mass[iP]*SpringConstant[Property[iP]]/(3.1415*3.1415+(log(AttenuationCoefficient[Property[iP]]))*(log(AttenuationCoefficient[Property[iP]]))));
                #pragma acc loop seq
                for(int iD=0;iD<DIM;iD++){
                force[iD] += -SpringConstant[Property[iP]]*(ParticleSpacing-rij)*eij[iD]-(DampingConstant*v_n[iD]*eij[iD])*Dt;
                }
              }
        }
            #pragma acc loop seq
         for (int iD=0;iD<DIM;iD++){
        Force[iP][iD] += force[iD];
                }
            }
        }
     }
     
        


static void calculateTangentialDirectionForce(){
    
        #pragma acc kernels present (Property[0:ParticleCount],RigidProperty[0:ParticleCount],Position[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
        #pragma acc loop independent
        #pragma omp parallel for
    for (int iP=0;iP<ParticleCount;iP++){
       if(Property[iP]>=FLUID_BEGIN && Property[iP]<FLUID_END) continue;
        double force[DIM] = {0.0,0.0,0.0};
      #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            double vij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            vij[iD] = Velocity[jP][iD]-Velocity[iP][iD];
            }
            
            double v_n[DIM];
              if(RigidProperty[iP] != RigidProperty[jP] && RigidProperty[iP]!= -1&& RigidProperty[iP]!= -1){
            if(rij2<=ParticleSpacing*ParticleSpacing){
                const double rij = sqrt(rij2);
                double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                
                #pragma acc loop seq
                for (int iD =0;iD<DIM;iD++){
                    v_n[iD] = vij[iD]*eij[iD];
                }
                double v_t[DIM] = {vij[0]-v_n[0]*eij[0],vij[1]-v_n[1]*eij[1],vij[2]-v_n[2]*eij[2]};
                double d_t[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
                for(int iD=0;iD<DIM;iD++){
                     d_t[iD] += v_t[iD]*Dt;
                }
              
                double spring_N[DIM];
                double spring_T[DIM];
                double abs_F_normal;
                double abs_F_tangen;
                const double DampingConstant = -2*log(AttenuationCoefficient[Property[iP]])*sqrt(Mass[iP]*SpringConstant[Property[iP]]/(3.1415*3.1415+(log(AttenuationCoefficient[Property[iP]]))*(log(AttenuationCoefficient[Property[iP]]))));
                #pragma acc loop seq
                for (int iD=0;iD<DIM;iD++){
                    spring_N[iD]=-SpringConstant[Property[iP]]*(ParticleSpacing-rij)*eij[iD]-(DampingConstant*v_n[iD]*eij[iD])*Dt;
                }
                abs_F_normal = FrictionCoefficient[Property[iP]]*sqrt(spring_N[0]*spring_N[0]+spring_N[1]*spring_N[1]+spring_N[2]*spring_N[2]);
                #pragma acc loop seq
                for (int iD=0;iD<DIM;iD++){
                    spring_T[iD]=-SpringConstant[Property[iP]]*d_t[iD]-(DampingConstant*v_t[iD]);
                }
                abs_F_tangen = sqrt(spring_T[0]*spring_T[0]+spring_T[1]*spring_T[1]+spring_T[2]*spring_T[2]);
                if( abs_F_normal > abs_F_tangen){
                    #pragma acc loop seq
                    for (int iD=0;iD<DIM;iD++){
                        force[iD] += -SpringConstant[Property[iP]]*d_t[iD]-(DampingConstant*v_t[iD])*Dt;
                    }
                }

                        else{
                            const double abs_v_t = sqrt(v_t[0]*v_t[0]+v_t[1]*v_t[1]+v_t[2]*v_t[2]);
                            if(abs_v_t != 0 ){
                            #pragma acc loop seq
                                for (int iD=0;iD<DIM;iD++){
                                force[iD] += abs_F_normal*v_t[iD]/abs_v_t;
                                }
                            }
                        }
                    }
                }
                #pragma acc loop seq
                for (int iD=0;iD<DIM;iD++){
              
                Force[iP][iD] += force[iD];
                }
		}
            }
            }
        





static void calculateEnergyConservation(){
    
#pragma acc kernels present(Temperature[0:ParticleCount],MeltingTemp[0:ParticleCount],Enthalpy[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount],Cp[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;iP++){
        if(Enthalpy[iP]<H0[iP]){
            Temperature[iP]=MeltingTemp[iP]+(Enthalpy[iP]-H0[iP])/Cp[iP];
        }
        else if(H0[iP]<=Enthalpy[iP] && Enthalpy[iP]<H1[iP]){
            Temperature[iP] =MeltingTemp[iP];
            
        }
        else if(Enthalpy[iP]>=H1[iP]) {
            Temperature[iP] = MeltingTemp[iP]+ (Enthalpy[iP]-H1[iP])/Cp[iP];
            
        }
    }
    
#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Temperature[0:ParticleCount],Conductivity[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Enthalpy[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double flux=0.0;
       #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
}
            
            const double radius = RadiusV;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 >= 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,radius);
                const double wwij = dwij/rij;
                flux += 2.0*Conductivity[iP]/(Density[Property[iP]])*(Temperature[jP]-Temperature[iP])*wwij*Dt;
            
            }
        }
        Enthalpy[iP] += flux;

}
}



static void calculateSolidFraction(){
    #pragma acc kernels present(Enthalpy[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;iP++){
        if(Enthalpy[iP]<H0[iP]){
            SolidFraction[iP] = 1.0;
        }
        else if(H0[iP]<=Enthalpy[iP] && Enthalpy[iP]< H1[iP]){
            SolidFraction[iP] = (H1[iP]-Enthalpy[iP])/(H1[iP]-H0[iP]);
        }
        else if(Enthalpy[iP]>=H1[iP]) {
            SolidFraction[iP] = 0.0;
        }
    }
     #pragma acc kernels present(Enthalpy[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=RigidBegin;iP<RigidEnd;iP++){
        if(Enthalpy[iP]<H0[iP]){
            SolidFraction[iP] = 1.0;
        }
        else if(H0[iP]<=Enthalpy[iP] && Enthalpy[iP]< H1[iP]){
            SolidFraction[iP] = (H1[iP]-Enthalpy[iP])/(H1[iP]-H0[iP]);
        }
        else if(Enthalpy[iP]>=H1[iP]) {
            SolidFraction[iP] = 0.0;
        }
    }
      #pragma acc kernels present(SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;iP++){
        SolidFraction[iP] = 0.0;
    }
}

static void calculateViscosity(){
    
    
 #pragma acc kernels present(SolidFraction[0:ParticleCount],Mu[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;iP++){
        if(SolidFraction[iP]<CriticalSolidFraction[Property[iP]]){
            Mu[iP] = ShearViscosity[Property[iP]]*exp(2.5*4.0*SolidFraction[iP]);
        }
        else {
            Mu[iP] = 100*ShearViscosity[Property[iP]]*exp(2.5*4.0*SolidFraction[iP]);
        }
    }
    
        #pragma acc kernels present(SolidFraction[0:ParticleCount],Mu[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=RigidBegin;iP<RigidEnd;iP++){
        if(SolidFraction[iP]<CriticalSolidFraction[Property[iP]]){
            Mu[iP] = ShearViscosity[Property[iP]]*exp(2.5*2.0*SolidFraction[iP]);
        }
        else {
            Mu[iP] = 100*ShearViscosity[Property[iP]]*exp(2.5*2.0*SolidFraction[iP]);
        }
    }
    

    

     #pragma acc kernels present(Mu[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;iP++){
        Mu[iP] = ShearViscosity[Property[iP]];
    }
}


static void calculateRadiation(){
  #pragma acc parallel loop \
        present(Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Temperature[0:ParticleCount], Enthalpy[0:ParticleCount], Mass[0:ParticleCount]) \
        present(NeighborCount[0:ParticleCount], DomainWidth[0:DIM], ParticleSpacing, Dt)
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const double TE = 650;
        double flux=0.0;
         if(NeighborCount[iP] < CUT && Temperature[iP] > Te){
          double T4 = Temperature[iP] * Temperature[iP] * Temperature[iP] * Temperature[iP];
          double Te4 = TE * TE * TE * TE*TE;

        flux += SBC * emissivity / Mass[iP] * ParticleSpacing * ParticleSpacing *Te4 * Dt;
    }
        Enthalpy[iP] -= flux;
    }
  }






static void calculateGravity(){
    #pragma acc kernels present (Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
    #pragma acc kernels present (Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=RigidBegin;iP<RigidEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
}

static void calculateAcceleration()
{
    #pragma acc kernels present (Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
    #pragma acc kernels present (Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=RigidBegin;iP<RigidEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
}


static void calculateWall()
{
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
    
    #pragma acc kernels present (Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){

     //   if(Time < 0.5){
        const int iProp = Property[iP];
        double r[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
        const double (&R)[DIM][DIM] = WallRotation[iProp];
        const double (&w)[DIM] = WallOmega[iProp];
        r[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
        r[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
        r[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];
        Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
        Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
        Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
        Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
        Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
        Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
      //  }
  
        
    }
    
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
        WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
        WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
        WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
    }
}

static int NonzeroCountA;
static double *CsrCofA;  // [ FluidCount * DIM x NeighFluidCount * DIM]
static int    *CsrIndA;  // [ FluidCount * DIM x NeighFluidCount * DIM]
static int    *CsrPtrA;  // [ FluidCount * DIM + 1 ] NeighborFluidCount\82̐ώZ\97\F1\82Ƃ\B5\82Čv\8EZ\89\C2
static double *VectorB;  // [ FluidCount * DIM ]
#pragma acc declare create(NonzeroCountA,CsrCofA,CsrIndA,CsrPtrA,VectorB)



static void calculateMatrixA( void )
{
    const double (*r)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
    const double (*m) = Mass;
    
    
    // Copy DIM*NeighborFluidCount to CsrPtrA
    int power = 0;
    const int fluidcount = (FluidParticleEnd-FluidParticleBegin)+(RigidEnd-RigidBegin);
    const int N = DIM*(fluidcount);
    while( (N>>power) != 0 ){
        power++;
    }
    const int powerN = (1<<power);
    
    CsrPtrA = (int *)malloc( powerN * sizeof(int));
    #pragma acc enter data create(CsrPtrA[0:powerN])
    
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<powerN;++iRow){
        CsrPtrA[iRow]=0;
    }

    #pragma acc kernels present(CsrPtrA[0:powerN])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD+1;
            CsrPtrA[iRow] = DIM*(NeighborFluidCount[iP]+NeighborRigidCount[iP]);
        }
    }
    
    // Convert CsrPtrA into cumulative sum
    for(int iMain=0;iMain<power;++iMain){
        const int dist = (1<<iMain);
        #pragma acc kernels present(CsrPtrA[0:powerN])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
            CsrPtrA[iRow]+=CsrPtrA[iRow+dist];
        }
    }
    for(int iMain=0;iMain<power;++iMain){
        const int dist = (powerN>>(iMain+1));
        #pragma acc kernels present(CsrPtrA[0:powerN])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
            CsrPtrA[iRow]-=CsrPtrA[iRow+dist];
            CsrPtrA[iRow+dist]+=CsrPtrA[iRow];
        }
    }
    
    #pragma acc kernels present(CsrPtrA[0:powerN])
    #pragma acc loop seq
    for(int iDummy=0;iDummy<1;++iDummy){
        NonzeroCountA=CsrPtrA[N];
    }
    #pragma acc update host(NonzeroCountA)
    
    // calculate coeeficient matrix A and source vector B
    CsrCofA = (double *)malloc( NonzeroCountA * sizeof(double) );
    CsrIndA = (int *)malloc( NonzeroCountA * sizeof(int) );
    VectorB = (double *)malloc( N * sizeof(double) );
    #pragma acc enter data create(CsrCofA[0:NonzeroCountA])
    #pragma acc enter data create(CsrIndA[0:NonzeroCountA])
    #pragma acc enter data create(VectorB[0:N])
    
    #pragma acc kernels present(CsrPtrA[0:N],CsrCofA[0:NonzeroCountA],CsrIndA[0:NonzeroCountA])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop independent
        for(int jN=0;jN<(NeighborFluidCount[iP]+NeighborRigidCount[iP]);++jN){
            const int jP = Neighbor[iP][jN];
            #pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                #pragma acc loop seq
                for(int sD=0;sD<DIM;++sD){
                    const int iRow    = DIM*iP+rD;
                    const int jColumn = DIM*jP+sD;
                    const int iNonzero = CsrPtrA[iRow]+DIM*jN+sD;
                    CsrIndA[ iNonzero ] = jColumn;
                }
            }
        }
    }
    
    #pragma acc kernels present(CsrCofA[0:NonzeroCountA])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iNonzero=0;iNonzero<NonzeroCountA;++iNonzero){
        CsrCofA[ iNonzero ] = 0.0;
    }
    
    #pragma acc kernels present(VectorB[0:N])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<N;++iRow){
        VectorB[iRow]=0.0;
    }
    
    #pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],m[0:ParticleCount],Mu[0:ParticleCount],CsrCofA[0:NonzeroCountA],CsrIndA[0:NonzeroCountA],CsrPtrA[0:N],VectorB[0:N])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        
        // Viscosity term
        int iN;
        double selfCof[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
        double sumvec[DIM]={0.0,0.0,0.0};
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP = Neighbor[iP][jN];
            if(iP==jP){
                iN=jN;
                continue;
            }
            double rij[DIM];
            #pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                rij[rD] =  Mod(Position[jP][rD] - Position[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
            }
            const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
            if(RadiusV*RadiusV -rij2 > 0){
                const double dij = sqrt(rij2);
                const double wdrij = -dwvdr(dij,RadiusV);
                const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
                const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
                
                #pragma acc loop seq
                for(int rD=0;rD<DIM;++rD){
                    const int iRow = DIM*iP+rD;
                    #pragma acc loop seq
                    for(int sD=0;sD<DIM;++sD){
                        #ifdef TWO_DIMENSIONAL
                        const double coefficient = +8.0*muij*eij[sD]*eij[rD]*wdrij/dij;
                        #else
                        const double coefficient = +10.0*muij*eij[sD]*eij[rD]*wdrij/dij;
                        #endif
                        
                        selfCof[rD][sD]+=coefficient;
                        
                        if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
                            const int jColumn = DIM*jP+sD;
                            const int jNonzero= CsrPtrA[iRow]+DIM*jN+sD;
                            // assert( CsrIndA [ jNonzero ] == jColumn);
                            CsrCofA [ jNonzero ] = -coefficient;
                        }
                        else if(RIGID_BEGIN<=Property[jP] && Property[jP]<RIGID_END){
                            const int jColumn = DIM*jP+sD;
                           const int jNonzero= CsrPtrA[iRow]+DIM*jN+sD;
                            // assert( CsrIndA [ jNonzero ] == jColumn);
                            CsrCofA [ jNonzero ] = -coefficient;
                        }
         	
                        else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                            sumvec[rD] += coefficient*Velocity[jP][sD];
                        }
                    
                    }
                }
            }
        }
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD;
            #pragma acc loop seq
            for(int sD=0;sD<DIM;++sD){
                const int iColumn = DIM*iP+sD;
                const int iNonzero= CsrPtrA[iRow]+DIM*iN+sD;
                // assert( CsrIndA[ iNonzero ] == iColumn);
                CsrCofA[ iNonzero ] = selfCof[rD][sD];
            }
        }
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD;
            VectorB[iRow]+=sumvec[rD];
        }
        
        // Ineritial Force term
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            const int iRow = DIM*iP+iD;
            const int iColumn = DIM*iP+iD;
            const int iNonzero= CsrPtrA[iRow]+DIM*iN+iD;
            const double coefficient =  m[iP]/ParticleVolume / Dt;
            // assert( CsrIndA[ iNonzero ] == iColumn );
            CsrCofA[ iNonzero ] += coefficient;
            VectorB[iRow] += coefficient*Velocity[iP][iD];
        }
    }
}

static int    NonzeroCountC;
static double *CsrCofC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrIndC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrPtrC; // [ FluidCount * DIM + 1 ] NeighCount\82̐ώZ\97\F1
static double *VectorP; // [ ParticleCount ]
#pragma acc declare create(NonzeroCountC, CsrCofC,CsrIndC,CsrPtrC,VectorP)

static void calculateMatrixC( void )
{
    const double (*r)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
    
    // Copy DIM*NeighborCountP to CsrPtrC
    int power = 0;
    const int fluidcount = (FluidParticleEnd-FluidParticleBegin)+(RigidEnd-RigidBegin);
    const int N = DIM*(fluidcount);
    while( (N>>power) != 0 ){
        power++;
    }
    const int powerN = (1<<power);
    
    CsrPtrC = (int *)malloc( powerN * sizeof(int));
    #pragma acc enter data create(CsrPtrC[0:powerN])
    
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<powerN;++iRow){
        CsrPtrC[iRow]=0;
    }
    
    #pragma acc kernels present(CsrPtrC[0:powerN])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow=DIM*iP+rD+1;
            CsrPtrC[iRow] = NeighborCountP[iP];
        }
    }
    
    // Convert CsrPtrC to cumulative sum
    for(int iMain=0;iMain<power;++iMain){
        const int dist = (1<<iMain);
        #pragma acc kernels present(CsrPtrC[0:powerN])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
            CsrPtrC[iRow]+=CsrPtrC[iRow+dist];
        }
    }
    for(int iMain=0;iMain<power;++iMain){
        const int dist = (powerN>>(iMain+1));
        #pragma acc kernels present(CsrPtrC[0:powerN])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
            CsrPtrC[iRow]-=CsrPtrC[iRow+dist];
            CsrPtrC[iRow+dist]+=CsrPtrC[iRow];
        }
    }
    
    #pragma acc kernels present(CsrPtrC[0:powerN])
    #pragma acc loop seq
    for(int iDummy=0;iDummy<1;++iDummy){
        NonzeroCountC = CsrPtrC[N];
    }
    #pragma acc update host(NonzeroCountC)
    
    // calculate coefficient matrix C and source vector P
    CsrCofC = (double *)malloc( NonzeroCountC * sizeof(double) );
    CsrIndC = (int *)malloc( NonzeroCountC * sizeof(int) );
    VectorP = (double *)malloc( ParticleCount * sizeof(double) );
    #pragma acc enter data create(CsrCofC[0:NonzeroCountC])
    #pragma acc enter data create(CsrIndC[0:NonzeroCountC])
    #pragma acc enter data create(VectorP[0:ParticleCount])
    
    #pragma acc kernels present(CsrPtrC[0:powerN],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],)
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop independent
        for(int jN=0;jN<NeighborCountP[iP];++jN){
            const int jP = NeighborP[iP][jN];
            #pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                const int iRow    = DIM*iP+rD;
                const int iColumn = jP;
                const int iNonzero = CsrPtrC[iRow]+jN;
                CsrIndC[ iNonzero ] = iColumn;
            }
        }
    }
    
    #pragma acc kernels present(CsrCofC[0:NonzeroCountC])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iNonzero=0;iNonzero<NonzeroCountC;++iNonzero){
        CsrCofC[ iNonzero ] = 0.0;
    }
    
    #pragma acc kernels present(VectorP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        VectorP[iP] = 0.0;
    }
    
    // set matrix C
    #pragma acc kernels present(r[0:ParticleCount][0:DIM],CsrCofC[0:NonzeroCountC],CsrIndC[0:NonzeroCountC],CsrPtrC[0:N])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        int iN;
        double selfCof[DIM] = {0.0,0.0,0.0};
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCountP[iP];++jN){
            const int jP = NeighborP[iP][jN];
            if(iP==jP){
           //     iN=jN;
                continue;
            }
            double rij[DIM];
            #pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                rij[rD] =  Mod(r[jP][rD] - r[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
            }
            const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
            if(RadiusP*RadiusP -rij2 > 0){
                const double dij = sqrt(rij2);
                const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
                const double wpdrij = -dwpdr(dij,RadiusP);
                #pragma acc loop seq
                for(int rD=0;rD<DIM;++rD){
                    const int iRow = DIM*iP+rD;
                    const double coefficient = eij[rD]*wpdrij;
                    selfCof[rD]+=coefficient;
                    const int jColumn = jP;
                    const int iNonzero = CsrPtrC[iRow]+jN;
                    //assert( CsrIndC[ iNonzero ] == jColumn );
                    CsrCofC[ iNonzero ] = coefficient;
                }
            }
        }
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD;
            const int iColumn = iP;
            const int iNonzero = CsrPtrC[iRow]+iN;
            //assert( CsrIndC[ iNonzero ] == iColumn );
            CsrCofC[ iNonzero ] = selfCof[rD];
        }
    }

    // set vector P
    #pragma acc kernels present(Property[0:ParticleCount],r[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Lambda[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Kappa[0:ParticleCount],VectorP[0:ParticleCount],VolStrainP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        
        VectorP[iP]=0.0;
        if(VolStrainP[iP]>0.0){
            VectorP[iP] = Kappa[iP]*VolStrainP[iP];
        }
        double sum = 0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCountP[iP];++jN){
            const int jP = NeighborP[iP][jN];
            if(iP==jP)continue;
            double rij[DIM];
#pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                rij[rD] =  Mod(r[jP][rD] - r[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
            }
            const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
            if(RadiusP*RadiusP -rij2 > 0){
                const double dij = sqrt(rij2);
                const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
                const double wpdrij = -dwpdr(dij,RadiusP);
#pragma acc loop seq
                for(int sD=0;sD<DIM;++sD){
                    if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
                        const double coefficient = +eij[sD]*wpdrij;
                        sum += Lambda[iP]*coefficient*v[iP][sD];
                    }
                    if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                        const double coefficient = -eij[sD]*wpdrij;
                        sum += Lambda[iP]*coefficient*v[jP][sD];
                    }
                }
            }
            
        }
        VectorP[iP]+=sum;
    }

}

static void multiplyMatrixC( void )
{
    const int fluidcount = (FluidParticleEnd-FluidParticleBegin)+(RigidEnd-RigidBegin);
    const int N = DIM*(fluidcount);
    const int M = ParticleCount;
    
    int power = 0;
    while( (N>>power) != 0 ){
        power++;
    }
    const int powerN = (1<<power);
    
    // b = b - Cp
    #pragma acc kernels present(CsrPtrC[0:powerN],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],VectorP[0:M])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<N;++iRow){
        #pragma acc loop seq
        for(int iNonzero=CsrPtrC[iRow];iNonzero<CsrPtrC[iRow+1];++iNonzero){
            const int iColumn = CsrIndC[iNonzero];
            VectorB[iRow] -= CsrCofC[iNonzero] * VectorP[iColumn];
        }
    }
    
    // A = A + C\83\A9C^T
    #pragma acc kernels present(CsrPtrA[0:N],CsrIndA[0:NonzeroCountA],CsrCofA[0:NonzeroCountA],CsrPtrC[0:N],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCountP[0:ParticleCount],Lambda[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop independent
        for(int jN=0;jN<NeighborFluidCount[iP]+NeighborRigidCount[iP];++jN){
            const int jP = Neighbor[iP][jN];
            int iNeigh=0;
            int jNeigh=0;
            double sum[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
            #pragma acc loop seq
            while(iNeigh<NeighborCountP[iP] && jNeigh<NeighborCountP[jP]){
                const int iNP = NeighborP[iP][iNeigh];
                const int jNP = NeighborP[jP][jNeigh];
                if(iNP==jNP){
                    #pragma acc loop seq
                    for(int rD=0;rD<DIM;++rD){
                        #pragma acc loop seq
                        for(int sD=0;sD<DIM;++sD){
                            const int iRowC    = DIM*iP+rD;
                            const int iColumnC = iNP;
                            const int iNonzeroC= CsrPtrC[iRowC]+iNeigh;
                            // assert(CsrIndC[iNonzeroC]==iColumnC);
                            const int jRowC    = DIM*jP+sD;
                            const int jColumnC = jNP;
                            const int jNonzeroC= CsrPtrC[jRowC]+jNeigh;
                            // assert(CsrIndC[jNonzeroC]==jColumnC);
                            sum[rD][sD] += CsrCofC[ iNonzeroC ] * Lambda[iNP] * CsrCofC[ jNonzeroC ];
                        }
                    }
                    iNeigh++;
                    jNeigh++;
                }
                else if(iNP<jNP){
                    iNeigh++;
                }
                else if(iNP>jNP){
                    jNeigh++;
                }
                //else{
                    //break;
                //}
            }
            #pragma acc loop seq
            for(int rD=0;rD<DIM;++rD){
                #pragma acc loop seq
                for(int sD=0;sD<DIM;++sD){
                    const int iRowA    = DIM*iP+rD;
                    const int iColumnA = DIM*jP+sD;
                    const int iNonzeroA= CsrPtrA[iRowA]+DIM*jN+sD;
                    // assert(CsrIndA[ iNonzeroA ]==iColumnA);
                    CsrCofA[iNonzeroA] += sum[rD][sD];
                }
            }
        }
    }
    
    free(CsrCofC);
    free(CsrIndC);
    free(CsrPtrC);
    free(VectorP);
    #pragma acc exit data delete(CsrPtrC,CsrIndC,CsrCofC,VectorP)

}




static void myDcsrmv( const int m, const int nnz, const double alpha, const double *csrVal, const int *csrRowPtr, const int *csrColInd, const double *x, const double beta, double *y)
{
    #pragma acc kernels present(csrVal[0:nnz],csrRowPtr[0:m+1],csrColInd[0:nnz],x[0:m],y[0:m])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<m; ++iRow){
        y[iRow] *= beta;
        #pragma acc loop seq
        for(int iNonZero=csrRowPtr[iRow];iNonZero<csrRowPtr[iRow+1];++iNonZero){
            const int iColumn=csrColInd[iNonZero];
            y[iRow] += alpha*csrVal[iNonZero]*x[iColumn];
        }
    }
}

static void myDdot( const int n, const double *x, const double *y, double *res )
{
    double sum=0.0;
    #pragma acc kernels copy(sum) present(x[0:n],y[0:n])
    #pragma acc loop reduction(+:sum)
    #pragma omp parallel for reduction(+:sum)
    for(int iRow=0;iRow<n;++iRow){
        sum += x[iRow]*y[iRow];
    }
    (*res)=sum;
}

static void myDaxpy( const int n, const double alpha, const double *x, double *y )
{
    #pragma acc kernels present(x[0:n],y[0:n])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<n;++iRow){
        y[iRow] += alpha*x[iRow];
    }
}

static void myDcopy( const int n, const double *x, double *y )
{
    #pragma acc kernels present(x[0:n],y[0:n])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iRow=0;iRow<n;++iRow){
        y[iRow] = x[iRow];
    }
}


static void solveWithConjugatedGradient(void){
    const int fluidcount = (FluidParticleEnd-FluidParticleBegin)+(RigidEnd-RigidBegin);
    const int N = DIM*fluidcount;
    
    const double *b = VectorB;
    double *x = (double *)malloc( N*sizeof(double) );
    double *r = (double *)malloc( N*sizeof(double) );
    double *z = (double *)malloc( N*sizeof(double) );
    double *p = (double *)malloc( N*sizeof(double) );
    double *q = (double *)malloc( N*sizeof(double) );
    double rho=0.0;
    double rhop=0.0;
    double tmp=0.0;
    double alpha=0.0;
    double beta=0.0;
    double nrm=0.0;
    double nrm0=0.0;
    int iter=0;
    
    #pragma acc enter data create(x[0:N],r[0:N],z[0:N],p[0:N],q[0:N])

    // intialize
    #pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],x[0:N],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD;
            x[iRow]=Velocity[iP][rD]-Force[iP][rD]/Mass[iP]*Dt;
        }
    }
        
    myDcopy( N, b, r );
    myDcsrmv( N, NonzeroCountA, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, r );
    myDdot( N, b, b, &nrm0 );
    nrm0=sqrt(nrm0);
    myDdot( N, r, r, &nrm );
    nrm=sqrt(nrm);
    
    if(nrm!=0.0)for(iter=0;iter<N;++iter){
        myDcopy( N, r, z );
        rhop = rho;
        myDdot( N, r, z, &rho);
        if(iter==0){
            myDcopy( N, z, p );
        }
        else{
            beta=rho/rhop;
            myDaxpy( N, beta, p, z );
            myDcopy( N, z, p );
        }
        myDcsrmv( N, NonzeroCountA, 1.0, CsrCofA, CsrPtrA, CsrIndA, p, 0.0, q);
        myDdot( N, p, q, &tmp );
        alpha =rho/tmp;
        myDaxpy( N, alpha, p, x );
        myDaxpy( N,-alpha, q, r );
        myDdot( N, r, r, &nrm );
        nrm=sqrt(nrm);
        
        if(nrm/nrm0 < 1.0e-7 )break;
        
    }
    
    log_printf("nrm=%e, nrm0=%e, iter=%d\n",nrm,nrm0,iter);
//    myDcopy( N, b, z );
//    myDcsrmv( N, N, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, z );
//    myDdot( N, z, z, &nrm );
//    nrm=sqrt(nrm);
//    fprintf(stderr,"check nrm=%e\n",nrm);
    
    
    //copy to Velocity
    #pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],x[0:N])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<fluidcount;++iP){
        #pragma acc loop seq
        for(int rD=0;rD<DIM;++rD){
            const int iRow = DIM*iP+rD;
            Velocity[iP][rD]=x[iRow];
        }
    }
    
    
    free(x);
    free(r);
    free(z);
    free(p);
    free(q);
    #pragma acc exit data delete(x[0:N],r[0:N],z[0:N],p[0:N],q[0:N])

    
    free(CsrCofA);
    free(CsrIndA);
    free(CsrPtrA);
    free(VectorB);
    #pragma acc exit data delete(CsrCofA,CsrIndA,CsrPtrA,VectorB)

        
}


   

static void calculateCenterofGravity() {
    #pragma acc kernels present (Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM], Mass[0:ParticleCount])
    {
        #pragma acc loop independent
        for (int iR = 0; iR < RigidPropertyCount; ++iR) {
            double rigidPosition[DIM] = {0.0, 0.0, 0.0};
            double rigidVelocity[DIM] = {0.0, 0.0, 0.0};
            double totalMass = 0.0;

            #pragma acc loop seq
            for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; ++iP) {
            #pragma acc loop seq
                for (int iD = 0; iD < DIM; ++iD) {
                    rigidPosition[iD] += Position[iP][iD] * Mass[iP];
                    rigidVelocity[iD] += Velocity[iP][iD] * Mass[iP];
                }
                totalMass += Mass[iP];
            }
	   #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                RigidPosition[iR][iD] = rigidPosition[iD] / totalMass;
                RigidVelocity[iR][iD] = rigidVelocity[iD] / totalMass;
            }

            double angularMomentum[DIM] = {0.0, 0.0, 0.0};

            #pragma acc loop seq
            for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; ++iP) {
                double dx = Position[iP][0] - RigidPosition[iR][0];
                double dy = Position[iP][1] - RigidPosition[iR][1];
                double dz = Position[iP][2] - RigidPosition[iR][2];

                angularMomentum[0] += Mass[iP] * (dy * Velocity[iP][2] - dz * Velocity[iP][1]);
                angularMomentum[1] += Mass[iP] * (dz * Velocity[iP][0] - dx * Velocity[iP][2]);
                angularMomentum[2] += Mass[iP] * (dx * Velocity[iP][1] - dy * Velocity[iP][0]);
            }
            #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                AngularMomentum[iR][iD] = angularMomentum[iD];
            }
        }
    }
}
static void calculateAngularMomentum() {
    #pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM], Mass[0:ParticleCount])
    for (int iR = 0; iR < RigidPropertyCount; iR++) {
        double I[DIM][DIM] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double r[DIM] = {0.0, 0.0, 0.0};

        // Calculate the inertia tensor for each rigid body
        #pragma acc loop seq
        for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; ++iP) {
            // Calculate the relative position of the particle w.r.t the rigid body's center of gravity
            #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                r[iD] = Position[iP][iD] - RigidPosition[iR][iD];
            }

            // Update inertia tensor components
            I[0][0] += Mass[iP] * (r[1] * r[1] + r[2] * r[2]);
            I[1][1] += Mass[iP] * (r[0] * r[0] + r[2] * r[2]);
            I[2][2] += Mass[iP] * (r[0] * r[0] + r[1] * r[1]);

            I[0][1] -= Mass[iP] * r[0] * r[1];
            I[0][2] -= Mass[iP] * r[0] * r[2];
            I[1][2] -= Mass[iP] * r[1] * r[2];

            I[1][0] = I[0][1];
            I[2][0] = I[0][2];
            I[2][1] = I[1][2];
        }

        // Compute the determinant of the inertia tensor
        double detI = I[0][0] * (I[1][1] * I[2][2] - I[1][2] * I[2][1])
                    - I[0][1] * (I[1][0] * I[2][2] - I[1][2] * I[2][0])
                    + I[0][2] * (I[1][0] * I[2][1] - I[1][1] * I[2][0]);

        // Compute the inverse of the inertia tensor if the determinant is non-zero
        if (detI != 0.0) {
            double invI[DIM][DIM];
            invI[0][0] = (I[1][1] * I[2][2] - I[1][2] * I[2][1]) / detI;
            invI[0][1] = -(I[0][1] * I[2][2] - I[0][2] * I[2][1]) / detI;
            invI[0][2] = (I[0][1] * I[1][2] - I[0][2] * I[1][1]) / detI;

            invI[1][0] = -(I[1][0] * I[2][2] - I[1][2] * I[2][0]) / detI;
            invI[1][1] = (I[0][0] * I[2][2] - I[0][2] * I[2][0]) / detI;
            invI[1][2] = -(I[0][0] * I[1][2] - I[1][0] * I[0][2]) / detI;

            invI[2][0] = (I[1][0] * I[2][1] - I[1][1] * I[2][0]) / detI;
            invI[2][1] = -(I[0][0] * I[2][1] - I[0][1] * I[2][0]) / detI;
            invI[2][2] = (I[0][0] * I[1][1] - I[0][1] * I[1][0]) / detI;

            // Store the inverse inertia tensor in the output array
            #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                #pragma acc loop seq
                for (int jD = 0; jD < DIM; ++jD) {
                    InertialTensorInverse[iR][iD][jD] = invI[iD][jD];
                }
            }
        }
    }
}


static void calculateAngularV() {
    #pragma acc kernels present (Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM], Force[0:ParticleCount][0:DIM])
    {
        #pragma acc loop independent
        for (int iR = 0; iR < RigidPropertyCount; ++iR) {
            // Preparation of Particle Position
            double s = Quaternion[iR][0];
            double vx = Quaternion[iR][1];
            double vy = Quaternion[iR][2];
            double vz = Quaternion[iR][3];

            double R[DIM][DIM];
            R[0][0] = 1 - 2 * vy * vy - 2 * vz * vz;
            R[0][1] = 2 * vx * vy - 2 * s * vz;
            R[0][2] = 2 * vx * vz + 2 * s * vy;
            R[1][0] = 2 * vx * vy + 2 * s * vz;
            R[1][1] = 1 - 2 * vx * vx - 2 * vz * vz;
            R[1][2] = 2 * vy * vz - 2 * s * vx;
            R[2][0] = 2 * vx * vz - 2 * s * vy;
            R[2][1] = 2 * vy * vz + 2 * s * vx;
            R[2][2] = 1 - 2 * vx * vx - 2 * vy * vy;

            // Calculation of Angular Momentum
            double torque[DIM] = {0.0, 0.0, 0.0};

            #pragma acc loop seq
            for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; iP++) {
                double r[DIM];
                double f[DIM];
                #pragma acc loop seq
                for (int iD = 0; iD < DIM; ++iD) {
                    r[iD] = Position[iP][iD] - RigidPosition[iR][iD];
                    f[iD] = Force[iP][iD];
                }

                torque[0] += r[1] * f[2] - r[2] * f[1];
                torque[1] += r[2] * f[0] - r[0] * f[2];
                torque[2] += r[0] * f[1] - r[1] * f[0];
            }
            #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                AngularMomentum[iR][iD] += torque[iD] * Dt;
            }

            // Calculation of Angular Velocity
            #pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                AngularV[iR][0] = 0.0;
                AngularV[iR][1] = 0.0;
                AngularV[iR][2] = 0.0;
                #pragma acc loop seq
                for (int jD = 0; jD < DIM; ++jD) {
                    AngularV[iR][iD] += InertialTensorInverse[iR][iD][jD] * AngularMomentum[iR][jD];
                }
            }

            // First, multiply R and InertialTensorInverse
            double Temp1[DIM][DIM] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

            #pragma acc loop collapse(3) independent
            for (int iD = 0; iD < DIM; ++iD) {
                for (int jD = 0; jD < DIM; ++jD) {
                    for (int kD = 0; kD < DIM; ++kD) {
                        Temp1[iD][jD] += R[iD][kD] * InertialTensorInverse[iR][kD][jD];
                    }
                }
            }

            // Then, multiply Temp1 and transposed R
            double Temp2[DIM][DIM] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

            #pragma acc loop collapse(3) independent
            for (int iD = 0; iD < DIM; ++iD) {
                for (int jD = 0; jD < DIM; ++jD) {
                    for (int kD = 0; kD < DIM; ++kD) {
                        Temp2[iD][jD] += Temp1[iD][kD] * R[kD][jD];  // Note the swapped indices for R
                    }
                }
            }

            // Now, update InertialTensorInverse with Temp2 values
            #pragma acc loop collapse(2) independent
            for (int iD = 0; iD < DIM; ++iD) {
                for (int jD = 0; jD < DIM; ++jD) {
                    InertialTensorInverse[iR][iD][jD] = Temp2[iD][jD];
                }
            }
        }
    }
}

static void updateRigidNumber() {
    #pragma acc kernels present (Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM],SolidFraction[0:ParticleCount],CriticalSolidFraction[0:ParticleCount],RigidProperty[0:ParticleCount])
    {
        #pragma acc loop independent
        for (int iR = 0; iR < RigidPropertyCount; ++iR) {
            #pragma acc loop seq
            for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; ++iP) {
                 if(SolidFraction[iP] < CriticalSolidFraction[Property[iP]]) {
                     RigidProperty[iP] = -1.0;
                 }
            }
        }
    }
}

static void calculateRigidBody() {
    #pragma acc kernels present (Property[0:ParticleCount], RigidProperty[0:ParticleCount],Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM], Force[0:ParticleCount][0:DIM], Mass[0:ParticleCount])
    {
        #pragma acc loop independent
        for (int iR = 0; iR < RigidPropertyCount; iR++) {
            #pragma acc loop seq
            for (int iP = RigidParticleBegin[iR]; iP < RigidParticleEnd[iR]; ++iP) {
            if(RigidProperty[iP]!= -1.0){
                Velocity[iP][0] = ((AngularV[iR][1] * (Position[iP][2] - RigidPosition[iR][2]) - AngularV[iR][2] * (Position[iP][1] - RigidPosition[iR][1])) + RigidVelocity[iR][0]);
                Velocity[iP][1] = ((AngularV[iR][2] * (Position[iP][0] - RigidPosition[iR][0]) - AngularV[iR][0] * (Position[iP][2] - RigidPosition[iR][2])) + RigidVelocity[iR][1]);
                Velocity[iP][2] = ((AngularV[iR][0] * (Position[iP][1] - RigidPosition[iR][1]) - AngularV[iR][1] * (Position[iP][0] - RigidPosition[iR][0])) + RigidVelocity[iR][2]);
          	  }
       	      }
   	 }
    }
}


static void updateQuaternion() {
    #pragma acc kernels present (Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM], \
                                 AngularV[0:RigidPropertyCount][0:DIM], Quaternion[0:RigidPropertyCount][0:4])
    {
        #pragma acc loop independent
        for (int iR = 0; iR < RigidPropertyCount; iR++) {
            // Step 1: Get the current quaternion and angular velocity
            double q0 = Quaternion[iR][0];
            double q1 = Quaternion[iR][1];
            double q2 = Quaternion[iR][2];
            double q3 = Quaternion[iR][3];
            
            double wx = AngularV[iR][0];
            double wy = AngularV[iR][1];
            double wz = AngularV[iR][2];

            double omega[4] = {0.0, wx, wy, wz};  // Angular velocity quaternion (w, x, y, z)

            // Compute k1
            double k1[4];
            k1[0] = 0.5 * (-q1 * wx - q2 * wy - q3 * wz);
            k1[1] = 0.5 * (q0 * wx + q2 * wz - q3 * wy);
            k1[2] = 0.5 * (q0 * wy - q1 * wz + q3 * wx);
            k1[3] = 0.5 * (q0 * wz + q1 * wy - q2 * wx);

            // Compute intermediate quaternion q + 0.5 * k1 * Dt
            double q1_mid[4] = {q0 + 0.5 * Dt * k1[0], q1 + 0.5 * Dt * k1[1], q2 + 0.5 * Dt * k1[2], q3 + 0.5 * Dt * k1[3]};

            // Compute k2
            double k2[4];
            k2[0] = 0.5 * (-q1_mid[1] * wx - q1_mid[2] * wy - q1_mid[3] * wz);
            k2[1] = 0.5 * (q1_mid[0] * wx + q1_mid[2] * wz - q1_mid[3] * wy);
            k2[2] = 0.5 * (q1_mid[0] * wy - q1_mid[1] * wz + q1_mid[3] * wx);
            k2[3] = 0.5 * (q1_mid[0] * wz + q1_mid[1] * wy - q1_mid[2] * wx);

            // Compute intermediate quaternion q + 0.5 * k2 * Dt
            double q2_mid[4] = {q0 + 0.5 * Dt * k2[0], q1 + 0.5 * Dt * k2[1], q2 + 0.5 * Dt * k2[2], q3 + 0.5 * Dt * k2[3]};

            // Compute k3
            double k3[4];
            k3[0] = 0.5 * (-q2_mid[1] * wx - q2_mid[2] * wy - q2_mid[3] * wz);
            k3[1] = 0.5 * (q2_mid[0] * wx + q2_mid[2] * wz - q2_mid[3] * wy);
            k3[2] = 0.5 * (q2_mid[0] * wy - q2_mid[1] * wz + q2_mid[3] * wx);
            k3[3] = 0.5 * (q2_mid[0] * wz + q2_mid[1] * wy - q2_mid[2] * wx);

            // Compute intermediate quaternion q + k3 * Dt
            double q3_mid[4] = {q0 + Dt * k3[0], q1 + Dt * k3[1], q2 + Dt * k3[2], q3 + Dt * k3[3]};

            // Compute k4
            double k4[4];
            k4[0] = 0.5 * (-q3_mid[1] * wx - q3_mid[2] * wy - q3_mid[3] * wz);
            k4[1] = 0.5 * (q3_mid[0] * wx + q3_mid[2] * wz - q3_mid[3] * wy);
            k4[2] = 0.5 * (q3_mid[0] * wy - q3_mid[1] * wz + q3_mid[3] * wx);
            k4[3] = 0.5 * (q3_mid[0] * wz + q3_mid[1] * wy - q3_mid[2] * wx);

            // Update the quaternion using the Runge-Kutta formula
            Quaternion[iR][0] += Dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
            Quaternion[iR][1] += Dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
            Quaternion[iR][2] += Dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0;
            Quaternion[iR][3] += Dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0;

            // Normalize the quaternion to maintain unit length
            double norm = sqrt(Quaternion[iR][0] * Quaternion[iR][0] + 
                               Quaternion[iR][1] * Quaternion[iR][1] + 
                               Quaternion[iR][2] * Quaternion[iR][2] + 
                               Quaternion[iR][3] * Quaternion[iR][3]);
            Quaternion[iR][0] /= norm;
            Quaternion[iR][1] /= norm;
            Quaternion[iR][2] /= norm;
            Quaternion[iR][3] /= norm;
        }
    }
}


static void calculateVirialPressureInsideRadius()
{
    const double (*x)[DIM] = Position;
    
    #pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
     for(int iP=0;iP<ParticleCount;++iP){
        int count=1;
        double sum = VirialPressureAtParticle[iP];
        #pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
            if(iP==jP)continue;
            if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(RadiusP*RadiusP - rij2 > 0){
                count +=1;
                sum += VirialPressureAtParticle[jP];
            }
        }
        VirialPressureInsideRadius[iP] = sum/count;
    }
}


static void calculateVirialStressAtParticle()
{
    const double (*x)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
    

    #pragma acc kernels present (VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            #pragma acc loop seq
            for(int jD=0;jD<DIM;++jD){
                VirialStressAtParticle[iP][iD][jD]=0.0;
            }
        }
    }
    
    #pragma acc kernels present(x[0:ParticleCount][0:DIM],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
        #pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
            // pressureP
            if(RadiusP*RadiusP - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = dwpdr(rij,RadiusP);
                double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #pragma acc loop seq
                    for(int jD=0;jD<DIM;++jD){
                        stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
                    }
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            #pragma acc loop seq
            for(int jD=0;jD<DIM;++jD){
                VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
            }
        }
    }
    
    #pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
       #pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
            
            // pressureA
            if(RadiusA*RadiusA - rij2 > 0){
                double ratio = InteractionRatio[Property[iP]][Property[jP]];
                const double rij = sqrt(rij2);
                const double dwij = ratio * dwadr(rij,RadiusA);
                double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
                }
               
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #pragma acc loop seq
                    for(int jD=0;jD<DIM;++jD){
                        stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
                    }
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            #pragma acc loop seq
            for(int jD=0;jD<DIM;++jD){
                VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
            }
        }

    }
    
    #pragma acc kernels present(x[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
       #pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
            
            // viscosity term
            if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
                const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
                const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #ifdef TWO_DIMENSIONAL
                    fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
                    #else
                    fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
                    #endif
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #pragma acc loop seq
                    for(int jD=0;jD<DIM;++jD){
                        stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
                    }
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            #pragma acc loop seq
            for(int jD=0;jD<DIM;++jD){
                VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
            }
        }
    }
    
    #pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
       #pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
            
            // diffuse interface force (1st term)
            if(RadiusG*RadiusG - rij2 > 0){
                const double a = CofA[Property[iP]]*(CofK)*(CofK);
                double ratio = InteractionRatio[Property[iP]][Property[jP]];
                const double rij = sqrt(rij2);
                const double weight = ratio * wg(rij,RadiusG);
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #pragma acc loop seq
                    for(int jD=0;jD<DIM;++jD){
                        stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
                    }
                }
            }
            
            // diffuse interface force (2nd term)
            if(RadiusG*RadiusG - rij2 > 0.0){
                const double a = CofA[Property[iP]]*(CofK)*(CofK);
                double ratio = InteractionRatio[Property[iP]][Property[jP]];
                const double rij = sqrt(rij2);
                const double dw = ratio * dwgdr(rij,RadiusG);
                const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
                double gr=0.0;
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    gr += (                     -GravityCenter[iP][iD])*xij[iD];
                }
                double fij[DIM] = {0.0,0.0,0.0};
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
                }
                #pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    #pragma acc loop seq
                    for(int jD=0;jD<DIM;++jD){
                        stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
                    }
                }
            }
        }
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            #pragma acc loop seq
            for(int jD=0;jD<DIM;++jD){
                VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
            }
        }
    }
    
    #pragma acc kernels present(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],VirialPressureAtParticle[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #ifdef TWO_DIMENSIONAL
        VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
        #else
        VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
        #endif
    }

}


static void calculatePeriodicBoundary( void )
{
    #pragma acc kernels present(Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}


