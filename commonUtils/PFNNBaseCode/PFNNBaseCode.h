#pragma once
#ifndef PFNN_BASE_CODE_H
#define PFNN_BASE_CODE_H

// TODO: move  code and leave only base definitions in the header
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <stdarg.h>
#include <glm/gtx/euler_angles.hpp>
#include <vector>
#include <Eigen/Dense>
#include <limits>
using namespace Eigen;

#define M_PI 3.14159265358979323846 // pi
const bool USE_STRAFE = false;
const float USE_RESPONSIVE_CHARACTER = 1.0f; // Or 0
const float UNITCONV_CM_TO_M = 0.01f;
const float UNITCONV_M_TO_CM = 100.0f;
#define USE_DIRECT_TARGET_MOTION

glm::vec3 mix_directions(glm::vec3 x, glm::vec3 y, float a);
glm::mat4 mix_transforms(glm::mat4 x, glm::mat4 y, float a);
glm::quat quat_exp(glm::vec3 l);
glm::vec2 segment_nearest(glm::vec2 v, glm::vec2 w, glm::vec2 p);

struct Options {

	bool invert_y;

	bool enable_ik;

	bool display_debug;
	bool display_debug_heights;
	bool display_debug_joints;
	bool display_debug_pfnn;
	bool display_hud_options;
	bool display_hud_stick;
	bool display_hud_speed;

	bool display_areas_jump;
	bool display_areas_walls;

	float display_scale;

	float extra_direction_smooth;
	float extra_velocity_smooth;
	float extra_strafe_smooth;
	float extra_crouched_smooth;
	float extra_gait_smooth;
	float extra_joint_smooth;

	Options()
		: invert_y(false)
		, enable_ik(true)
		, display_debug(true)
		, display_debug_heights(true)
		, display_debug_joints(false)
		, display_debug_pfnn(false)
		, display_hud_options(true)
		, display_hud_stick(true)
		, display_hud_speed(true)
		, display_areas_jump(false)
		, display_areas_walls(false)
#ifdef HIGH_QUALITY
		, display_scale(3.0)
#else
		, display_scale(2.0)
#endif
		, extra_direction_smooth(1.0f)
		, extra_velocity_smooth(1.0f)
		, extra_strafe_smooth(0.9f)
		, extra_crouched_smooth(0.9f)
		, extra_gait_smooth(0.1f)
		, extra_joint_smooth(0.5f)
	{}
};

/* Character def - pose, joints, options etc*/
struct Character
{

	static const int JOINT_NUM = 31;

	//GLuint vbo, tbo;
	int ntri, nvtx;
	float phase;
	float prev_phase;
	float strafe_amount;
	float strafe_target;
	float crouched_amount;
	float crouched_target;
	float responsive;

	// Always use this function to set the desired maximum speed for your actor
	// In centimeters per second
	void setDesiredSpeed(const float desiredSpeed_cm);

	glm::vec3 joint_positions[JOINT_NUM];
	glm::vec3 joint_velocities[JOINT_NUM];
	glm::mat3 joint_rotations[JOINT_NUM];

	glm::mat4 joint_anim_xform[JOINT_NUM];
	glm::mat4 joint_rest_xform[JOINT_NUM];
	glm::mat4 joint_mesh_xform[JOINT_NUM];
	glm::mat4 joint_global_rest_xform[JOINT_NUM];
	glm::mat4 joint_global_anim_xform[JOINT_NUM];

	std::vector<int> joint_parents;

	int speedConverter_degree = 0;
	float* speedConverter_coefficients = nullptr;

	enum {
		JOINT_ROOT_L = 1,
		JOINT_HIP_L = 2,
		JOINT_KNEE_L = 3,
		JOINT_HEEL_L = 4,
		JOINT_TOE_L = 5,

		JOINT_ROOT_R = 6,
		JOINT_HIP_R = 7,
		JOINT_KNEE_R = 8,
		JOINT_HEEL_R = 9,
		JOINT_TOE_R = 10
	};

	Character()
		: /*vbo(0)
		, tbo(0)
		, */ntri(66918)
		, nvtx(11200)
		, phase(0)
		, prev_phase(0)
		, strafe_amount(0)
		, strafe_target(0)
		, crouched_amount(0)
		, crouched_target(0)
		, responsive(USE_RESPONSIVE_CHARACTER) 
		{
			joint_parents.resize(JOINT_NUM);
		}

	~Character() {
		//if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
		//if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }

		delete[] speedConverter_coefficients;
	}


	float bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax, bbox_zmin, bbox_zmax;
	void getBboxSize(float& xsize, float& ysize, float& zsize)
	{
		xsize = bbox_xmax - bbox_xmin;
		ysize = bbox_ymax - bbox_ymin;
		zsize = bbox_zmax - bbox_zmin;
	}

	void load(const char* filename_v, const char* filename_t, const char* filename_p, const char* filename_r, const char* filename_speedConverter) {

		printf("Read Character '%s %s'\n", filename_v, filename_t);

		//if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
		//if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }

		//glGenBuffers(1, &vbo);
		//glGenBuffers(1, &tbo);

		FILE *f;

		f = fopen(filename_v, "rb");
		float *vbo_data = (float*)malloc(sizeof(float) * 15 * nvtx);
		fread(vbo_data, sizeof(float) * 15 * nvtx, 1, f);
		fclose(f);

		f = fopen(filename_t, "rb");
		uint32_t *tbo_data = (uint32_t*)malloc(sizeof(uint32_t) * ntri);
		fread(tbo_data, sizeof(uint32_t) * ntri, 1, f);
		fclose(f);

		//glBindBuffer(GL_ARRAY_BUFFER, vbo);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 15 * nvtx, vbo_data, GL_STATIC_DRAW);

		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tbo);
		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * ntri, tbo_data, GL_STATIC_DRAW);

		free(vbo_data);
		free(tbo_data);

		f = fopen(filename_p, "rb");
		float fparents[JOINT_NUM];
		fread(fparents, sizeof(float) * JOINT_NUM, 1, f);
		for (int i = 0; i < JOINT_NUM; i++) { joint_parents[i] = (int)fparents[i]; }
		fclose(f);

		f = fopen(filename_r, "rb");
		fread(glm::value_ptr(joint_rest_xform[0]), sizeof(float) * JOINT_NUM * 4 * 4, 1, f);
		for (int i = 0; i < JOINT_NUM; i++) { joint_rest_xform[i] = glm::transpose(joint_rest_xform[i]); }
		fclose(f);

		// Read speed converter function
		f = fopen(filename_speedConverter, "rb");
		fread(&speedConverter_degree, sizeof(int), 1, f);
		speedConverter_coefficients = new float[speedConverter_degree];
		fread(speedConverter_coefficients, sizeof(float), speedConverter_degree, f);
		fclose(f);
	}

	float m_INPUT_SPEED = 0.0f;
	float m_MAX_SPEED = 0.0f; // max speed achieved by the model at this moment

	void forward_kinematics() 
	{
		for (int i = 0; i < JOINT_NUM; i++) {
			joint_global_anim_xform[i] = joint_anim_xform[i];
			joint_global_rest_xform[i] = joint_rest_xform[i];
			int j = joint_parents[i];
			while (j != -1) {
				joint_global_anim_xform[i] = joint_anim_xform[j] * joint_global_anim_xform[i];
				joint_global_rest_xform[i] = joint_rest_xform[j] * joint_global_rest_xform[i];
				j = joint_parents[j];
			}
			joint_mesh_xform[i] = joint_global_anim_xform[i] * glm::inverse(joint_global_rest_xform[i]);
		}

	}
};

struct PFNN {

	enum { XDIM = 342, YDIM = 311, HDIM = 512 };
	enum { MODE_CONSTANT, MODE_LINEAR, MODE_CUBIC };

	int mode;

	ArrayXf Xmean, Xstd;
	ArrayXf Ymean, Ystd;

	std::vector<ArrayXXf> W0, W1, W2;
	std::vector<ArrayXf>  b0, b1, b2;

	ArrayXf  Xp, Yp;
	ArrayXf  H0, H1;
	ArrayXXf W0p, W1p, W2p;
	ArrayXf  b0p, b1p, b2p;

	PFNN(int pfnnmode)
		: mode(pfnnmode) {

		Xp = ArrayXf((int)XDIM);
		Yp = ArrayXf((int)YDIM);

		H0 = ArrayXf((int)HDIM);
		H1 = ArrayXf((int)HDIM);

		W0p = ArrayXXf((int)HDIM, (int)XDIM);
		W1p = ArrayXXf((int)HDIM, (int)HDIM);
		W2p = ArrayXXf((int)YDIM, (int)HDIM);

		b0p = ArrayXf((int)HDIM);
		b1p = ArrayXf((int)HDIM);
		b2p = ArrayXf((int)YDIM);
	}

	static void load_weights(ArrayXXf &A, int rows, int cols, const std::string& basePath, const char* fmt, ...) {
		va_list valist;
		va_start(valist, fmt);
		char filename[512];
		vsprintf(filename, fmt, valist);
		va_end(valist);

		std::string fullPath = basePath + std::string(filename);

		FILE *f = fopen(fullPath.c_str(), "rb");
		if (f == NULL)
		{
			fprintf(stderr, "Couldn't load file %s\n", fullPath.c_str());
			exit(1);
		}



#if 1
		A = ArrayXXf(cols, rows);
		/*    for (int x = 0; x < rows; x++)
			{

				float* beginOfLineX = &A(x, 0);
				float* nextOfLineX = &A(x, 1);
				float* myEstBeginOfLineX = A.data() + x * cols * sizeof(float);

				fread(beginOfLineX, sizeof(float), cols, f);
			}
			*/
		fread(A.data(), sizeof(float)*rows * cols, 1, f);

		A.transposeInPlace();
#else 
		A = ArrayXXf(rows, cols);

		float* a = new float[rows * cols];
		fread(a, sizeof(float) * rows * cols, 1, f);
		rewind(f);

		for (int x = 0; x < rows; x++)
			for (int y = 0; y < cols; y++)
			{
				const float valueInArray = a[x * cols + y];
#if 0
				float item = 0;
				fread(&item, sizeof(float), 1, f);


				if (item != valueInArray)
				{
					int a = 3;
					a++;
				}
#endif
				A(x, y) = valueInArray;
			}
#endif

		fclose(f);
	}

	static void load_weights(ArrayXf &V, int items, const std::string& basePath, const char* fmt, ...) {
		va_list valist;
		va_start(valist, fmt);
		char filename[512];
		vsprintf(filename, fmt, valist);
		va_end(valist);

		std::string fullPath = basePath + std::string(filename);

		FILE *f = fopen(fullPath.c_str(), "rb");
		if (f == NULL)
		{
			fprintf(stderr, "Couldn't load file %s\n",
				fullPath.c_str());
			exit(1);
		}

		V = ArrayXf(items);

#if 1
		fread(V.data(), sizeof(float) * items, 1, f);
#else
		for (int i = 0; i < items; i++) {
			float item = 0.0;
			fread(&item, sizeof(float), 1, f);
			V(i) = item;
		}
#endif
		fclose(f);
	}

	void load(const std::string& strAssetsDir)
	{

		load_weights(Xmean, XDIM, strAssetsDir, "Parameters/pfnn/Xmean.bin");
		load_weights(Xstd, XDIM, strAssetsDir, "Parameters/pfnn/Xstd.bin");
		load_weights(Ymean, YDIM, strAssetsDir, "Parameters/pfnn/Ymean.bin");
		load_weights(Ystd, YDIM, strAssetsDir, "Parameters/pfnn/Ystd.bin");

		switch (mode) {

		case MODE_CONSTANT:

			W0.resize(50); W1.resize(50); W2.resize(50);
			b0.resize(50); b1.resize(50); b2.resize(50);

			for (int i = 0; i < 50; i++) {
				load_weights(W0[i], HDIM, XDIM, strAssetsDir, "Parameters/pfnn/W0_%03i.bin", i);
				load_weights(W1[i], HDIM, HDIM, strAssetsDir, "Parameters/pfnn/W1_%03i.bin", i);
				load_weights(W2[i], YDIM, HDIM, strAssetsDir, "Parameters/pfnn/W2_%03i.bin", i);
				load_weights(b0[i], HDIM, strAssetsDir, "Parameters/pfnn/b0_%03i.bin", i);
				load_weights(b1[i], HDIM, strAssetsDir, "Parameters/pfnn/b1_%03i.bin", i);
				load_weights(b2[i], YDIM, strAssetsDir, "Parameters/pfnn/b2_%03i.bin", i);
			}

			break;
			/*
		case MODE_LINEAR:

			W0.resize(10); W1.resize(10); W2.resize(10);
			b0.resize(10); b1.resize(10); b2.resize(10);

			for (int i = 0; i < 10; i++) {
				load_weights(W0[i], HDIM, XDIM, (strAssetsDir + std::string("/network/pfnn/W0_%03i.bin", i * 5)).c_str());
				load_weights(W1[i], HDIM, HDIM, (strAssetsDir + std::string("/network/pfnn/W1_%03i.bin", i * 5)).c_str());
				load_weights(W2[i], YDIM, HDIM, (strAssetsDir + std::string("/network/pfnn/W2_%03i.bin", i * 5)).c_str());
				load_weights(b0[i], HDIM, (strAssetsDir + std::string("/network/pfnn/b0_%03i.bin", i * 5)).c_str());
				load_weights(b1[i], HDIM, (strAssetsDir + std::string("/network/pfnn/b1_%03i.bin", i * 5)).c_str());
				load_weights(b2[i], YDIM, (strAssetsDir + std::string("/network/pfnn/b2_%03i.bin", i * 5)).c_str());
			}

			break;

		case MODE_CUBIC:

			W0.resize(4); W1.resize(4); W2.resize(4);
			b0.resize(4); b1.resize(4); b2.resize(4);

			for (int i = 0; i < 4; i++) {
				load_weights(W0[i], HDIM, XDIM, (strAssetsDir + std::string("/network/pfnn/W0_%03i.bin", (int)(i * 12.5))).c_str());
				load_weights(W1[i], HDIM, HDIM, (strAssetsDir + std::string("/network/pfnn/W1_%03i.bin", (int)(i * 12.5))).c_str());
				load_weights(W2[i], YDIM, HDIM, (strAssetsDir + std::string("/network/pfnn/W2_%03i.bin", (int)(i * 12.5))).c_str());
				load_weights(b0[i], HDIM, (strAssetsDir + std::string("/network/pfnn/b0_%03i.bin", (int)(i * 12.5))).c_str());
				load_weights(b1[i], HDIM, (strAssetsDir + std::string("/network/pfnn/b1_%03i.bin", (int)(i * 12.5))).c_str());
				load_weights(b2[i], YDIM, (strAssetsDir + std::string("/network/pfnn/b2_%03i.bin", (int)(i * 12.5))).c_str());
			}

			break;
			*/
		}
	}

	static void ELU(ArrayXf &x) { x = x.max(0) + x.min(0).exp() - 1; }

	static void linear(ArrayXf  &o, const ArrayXf  &y0, const ArrayXf  &y1, float mu) { o = (1.0f - mu) * y0 + (mu)* y1; }
	static void linear(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, float mu) { o = (1.0f - mu) * y0 + (mu)* y1; }

	static void cubic(ArrayXf  &o, const ArrayXf &y0, const ArrayXf &y1, const ArrayXf &y2, const ArrayXf &y3, float mu) {
		o = (
			(-0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3)*mu*mu*mu +
			(y0 - 2.5*y1 + 2.0*y2 - 0.5*y3)*mu*mu +
			(-0.5*y0 + 0.5*y2)*mu +
			(y1));
	}

	static void cubic(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, const ArrayXXf &y2, const ArrayXXf &y3, float mu) {
		o = (
			(-0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3)*mu*mu*mu +
			(y0 - 2.5*y1 + 2.0*y2 - 0.5*y3)*mu*mu +
			(-0.5*y0 + 0.5*y2)*mu +
			(y1));
	}

	void predict(float P) {

		float pamount;
		int pindex_0, pindex_1, pindex_2, pindex_3;

		Xp = (Xp - Xmean) / Xstd;

		switch (mode) {

		case MODE_CONSTANT:
			pindex_1 = (int)((P / (2 * M_PI)) * 50);
			H0 = (W0[pindex_1].matrix() * Xp.matrix()).array() + b0[pindex_1]; ELU(H0);
			H1 = (W1[pindex_1].matrix() * H0.matrix()).array() + b1[pindex_1]; ELU(H1);
			Yp = (W2[pindex_1].matrix() * H1.matrix()).array() + b2[pindex_1];
			break;

		case MODE_LINEAR:
			pamount = fmod((P / (2 * M_PI)) * 10, 1.0);
			pindex_1 = (int)((P / (2 * M_PI)) * 10);
			pindex_2 = ((pindex_1 + 1) % 10);
			linear(W0p, W0[pindex_1], W0[pindex_2], pamount);
			linear(W1p, W1[pindex_1], W1[pindex_2], pamount);
			linear(W2p, W2[pindex_1], W2[pindex_2], pamount);
			linear(b0p, b0[pindex_1], b0[pindex_2], pamount);
			linear(b1p, b1[pindex_1], b1[pindex_2], pamount);
			linear(b2p, b2[pindex_1], b2[pindex_2], pamount);
			H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
			H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
			Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
			break;

		case MODE_CUBIC:
			pamount = fmod((P / (2 * M_PI)) * 4, 1.0);
			pindex_1 = (int)((P / (2 * M_PI)) * 4);
			pindex_0 = ((pindex_1 + 3) % 4);
			pindex_2 = ((pindex_1 + 1) % 4);
			pindex_3 = ((pindex_1 + 2) % 4);
			cubic(W0p, W0[pindex_0], W0[pindex_1], W0[pindex_2], W0[pindex_3], pamount);
			cubic(W1p, W1[pindex_0], W1[pindex_1], W1[pindex_2], W1[pindex_3], pamount);
			cubic(W2p, W2[pindex_0], W2[pindex_1], W2[pindex_2], W2[pindex_3], pamount);
			cubic(b0p, b0[pindex_0], b0[pindex_1], b0[pindex_2], b0[pindex_3], pamount);
			cubic(b1p, b1[pindex_0], b1[pindex_1], b1[pindex_2], b1[pindex_3], pamount);
			cubic(b2p, b2[pindex_0], b2[pindex_1], b2[pindex_2], b2[pindex_3], pamount);
			H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
			H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
			Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
			break;

		default:
			break;
		}

		Yp = (Yp * Ystd) + Ymean;

	}
};


// For a sample recoreded, output some pose metrics
struct SamplePoseStats
{
	// About velocities, unit of measure: cm/s 
	float min_vel[Character::JOINT_NUM];
	float max_vel[Character::JOINT_NUM];
	float avg_vel[Character::JOINT_NUM];

	// About rotations on Yaw: degrees
	float max_yaw[Character::JOINT_NUM];
	float avg_yaw[Character::JOINT_NUM];

	float maximum_max_yaws;
	float avg_max_yaws;
	float maximum_avg_yaws;
	float avg_avg_yaws;

	void reset()
	{
		for (int i = 0; i < Character::JOINT_NUM; i++)
		{
			min_vel[i] = std::numeric_limits<float>::max();
			max_vel[i] = std::numeric_limits<float>::min();
			avg_vel[i] = 0.0f;
			
			avg_yaw[i] = 0.0f;
			max_yaw[i] = std::numeric_limits<float>::min();

			float maximum_max_yaws = std::numeric_limits<float>::min();
			float avg_max_yaws = 0.0f;
			float maximum_avg_yaws = std::numeric_limits<float>::min();
			float avg_avg_yaws = 0.0f;
		}
	}
};


/* Trajectory */

struct Trajectory {

	enum { LENGTH = 120 , SAMPLE_FACTOR = 10};



	float width;

	glm::vec3 positions[LENGTH];
	glm::vec3 directions[LENGTH];
	glm::mat3 rotations[LENGTH];
	float heights[LENGTH];

	float gait_stand[LENGTH];
	float gait_walk[LENGTH];
	float gait_jog[LENGTH];
	float gait_crouch[LENGTH];
	float gait_jump[LENGTH];
	float gait_bump[LENGTH];

	glm::vec3 target_dir, target_vel;

	Trajectory()
		: width(25)
		, target_dir(glm::vec3(0, 0, 1))
		, target_vel(glm::vec3(0)) {}

};


struct PFNNCharacterBase
{
public:
	PFNNCharacterBase() { m_name = "no_name"; }
	PFNNCharacterBase(const std::string &name) { m_name = name; }
	// Initializes the character by given the assets folder (network weights, skeleton config)
	void init(const std::string& strAssetsDir);

	// Resets the position of this character to the specified pos
	void resetPosAndOrientation(glm::vec2 position, glm::vec2 orientation);

	// Set the target position to go of the character
	void setTargetPosition(glm::vec3 targetPos) 
	{ 
		g_nextTargetToGo = targetPos; 
	}

	// Update and post update 
	void updateAnim(float deltaTime);
	void postUpdateAnim(float deltaTime);

	// Get number of joints, joint pos and parent of each joint index
	////////////
	int getNumJoints() const
	{
		return Character::JOINT_NUM;
	}

	glm::vec3 getJointPos(int i) const
	{
		return character->joint_positions[i];
	}

	int getJointParent(int i) const
	{
		return character->joint_parents[i];
	}

	const std::string& getName() const 
	{
		return m_name;
	}

	// Set the desired speed in cm per second
	// Always use this function to set the desired maximum speed for your actor
	void setDesiredSpeed(float desiredSpeed_cm)
	{
		character->setDesiredSpeed(desiredSpeed_cm);
	}

	// Get trajectory len, sample factor and position of each point
	///////
	int getTrajectoryLen() const
	{
		return Trajectory::LENGTH;
	}

	int getTrajectorySampleFactor() const
	{
		return Trajectory::SAMPLE_FACTOR;
	}

	const glm::vec3 getTrajectoryPos(int i) const
	{
		return trajectory->positions[i];
	}
	//////

	// Get the current average speed of the character
	float  getCurrentAvgSpeed()
	{
		return m_currentSpeed;
	}

	// Get the current pose stats of the character
	const SamplePoseStats& getCurrentPoseStats() const
	{
		return m_currentPoseStats;
	}

	// Get the speed set internally in the system
	const float getInternalSystemSpeed()
	{
		return character->m_INPUT_SPEED;
	}

	// Get the destination goal distance threshold
	float getTargetReachedThreshold() const
	{
		return m_targetReachedThreshold;
	}

	// Try to not use this function and let the default run :)
	void setTargetReachedThreshold(const float newThreshold)
	{
		m_targetReachedThreshold = newThreshold;
	}

//private:

	glm::vec3 g_nextTargetToGo = glm::vec3(0.0f, 0.0f, 0.0f);
	Trajectory* trajectory = nullptr;
	PFNN* pfnn = nullptr;
	Options* options = nullptr;
	std::string m_name;

	float m_currentSpeed = 0.0f;
	SamplePoseStats m_currentPoseStats;

	Character* character = nullptr;
	float m_targetReachedThreshold = 30.0f;

	struct SpeedSample 
	{ 
		glm::vec3 pos; 
		float time = -1.0f; 
	};

	struct PoseSample
	{
		glm::vec3 jointPos[Character::JOINT_NUM];
		float yaw[Character::JOINT_NUM];
		float time = -1.0f;
	};

	// Variables for computing SPEED stats

	float m_currentSimTime = 0.0f;
	bool m_lastRootPosValid = false;
	glm::vec3 m_lastRootPos = glm::vec3(0.0f);
	glm::vec3 m_currentRootPos = glm::vec3(0.0f);

	static const int g_numSpeedSamples = 30;
	SpeedSample m_speedSamples[g_numSpeedSamples];
	int m_sampleSpeedHeadIndex = 0;

	static const int g_numPoseSamples = 30;
	PoseSample m_poseSamples[g_numPoseSamples];
	int m_samplePoseHeadIndex = 0;
	
	void addCurrentRootSample(glm::vec3 newPos, float deltaTime) 
	{ 
		m_speedSamples[m_sampleSpeedHeadIndex].pos = newPos;
		m_speedSamples[m_sampleSpeedHeadIndex].time = deltaTime;
		m_sampleSpeedHeadIndex++;

		if (m_sampleSpeedHeadIndex >= g_numSpeedSamples)
		{
			computeCurrentSpeed();
		}
	}

	void addCurrentPoseSample(const glm::vec3* jointPositions, const glm::mat4* transform44, const float deltaTime)
	{
		// Copy the pose to current head index...
		for (int i = 0; i < Character::JOINT_NUM; i++)
		{
			m_poseSamples[m_samplePoseHeadIndex].jointPos[i] = jointPositions[i];


			float rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f;
			glm::extractEulerAngleXYZ(transform44[i], rotX, rotY, rotZ);
			const float rotYinDeg = glm::degrees(rotY);
			m_poseSamples[m_samplePoseHeadIndex].yaw[i] = rotYinDeg;
		}
		m_poseSamples[m_samplePoseHeadIndex].time = deltaTime;

		m_samplePoseHeadIndex++;
		if (m_samplePoseHeadIndex >= g_numPoseSamples)
		{
			computeSamplePoseStats();
		}
	}

	// This will give you the current speed of the actor
	void computeCurrentSpeed()
	{
		float distanceTravelled = 0.0f;
		float totalTime = 0.0f;
		for (int i = 0; i < g_numSpeedSamples - 1; i++)
		{
			const float dist = glm::distance(m_speedSamples[i + 1].pos, m_speedSamples[i].pos);
			const float time = m_speedSamples[i + 1].time - m_speedSamples[i].time;

			distanceTravelled += dist;
			totalTime += time;
		}

		m_currentSpeed = distanceTravelled / totalTime;
		m_sampleSpeedHeadIndex = 0;

		// Update statistics about max speed
		if (character->m_MAX_SPEED < m_currentSpeed)
		{
			character->m_MAX_SPEED = m_currentSpeed;
		}
	}

	// Compute stats such as average, min, max displacement during the sample for each of the bones
	void computeSamplePoseStats()
	{
		m_currentPoseStats.reset();

		for (int frameIndex = 0; frameIndex < g_numSpeedSamples - 1; frameIndex++)
		{
			const PoseSample& currPoseSample = m_poseSamples[frameIndex];
			const PoseSample& nextPoseSample = m_poseSamples[frameIndex + 1];
			
			for (int boneIndex = 0; boneIndex < Character::JOINT_NUM; boneIndex++)
			{
				// Get distance in time between each bone movement between frames
				// Then 	compute instant velocity and update stats per bone
				const float dist = glm::distance(currPoseSample.jointPos[boneIndex], nextPoseSample.jointPos[boneIndex]);
				const float time = nextPoseSample.time - currPoseSample.time;
				const float boneVel = time == 0.0 ? 0.0f : (dist / time);
				m_currentPoseStats.avg_vel[boneIndex] += boneVel;
				m_currentPoseStats.min_vel[boneIndex] = std::min(m_currentPoseStats.min_vel[boneIndex], boneVel);
				m_currentPoseStats.max_vel[boneIndex] = std::max(m_currentPoseStats.max_vel[boneIndex], boneVel);


				const float yawCurrent = currPoseSample.yaw[boneIndex];
				const float yawNext = nextPoseSample.yaw[boneIndex];
				float phi = (yawCurrent - yawNext);
				if (phi > 360.0f)
					phi = phi - 360.0f;
				int res = phi > 180 ? 360 - phi : phi;
				const int sign = (yawCurrent - yawNext >= 0 && yawCurrent - yawNext <= 180) || (yawCurrent - yawNext <= -180 && yawCurrent - yawNext >= -360) ? 1 : -1;
				res *= sign;				
				const float angleDiff = res;

				m_currentPoseStats.max_yaw[boneIndex] = std::max(m_currentPoseStats.max_yaw[boneIndex], angleDiff);
				m_currentPoseStats.avg_yaw[boneIndex] += angleDiff;				
			}
		}

		const int numDiffSamplesTaken = (g_numSpeedSamples - 1);
		for (int boneIndex = 0; boneIndex < Character::JOINT_NUM; boneIndex++)
		{
			m_currentPoseStats.avg_vel[boneIndex] /= numDiffSamplesTaken;
			m_currentPoseStats.avg_yaw[boneIndex] /= numDiffSamplesTaken;
		}

		// COmpute some yaw stats for the first 10 bones:
		int NUM_BONES_FOR_YAW_STAT = 10;
		m_currentPoseStats.avg_max_yaws = 0.0f;
		m_currentPoseStats.avg_avg_yaws = 0.0f;
		m_currentPoseStats.maximum_max_yaws = -1.0f;
		m_currentPoseStats.maximum_avg_yaws = -1.0f;

		for (int boneIndex = 1; boneIndex <= NUM_BONES_FOR_YAW_STAT; boneIndex++)
		{
			m_currentPoseStats.maximum_max_yaws = std::max(m_currentPoseStats.maximum_max_yaws, m_currentPoseStats.max_yaw[boneIndex]);
			m_currentPoseStats.maximum_avg_yaws = std::max(m_currentPoseStats.maximum_avg_yaws, m_currentPoseStats.avg_yaw[boneIndex]);

			m_currentPoseStats.avg_max_yaws += m_currentPoseStats.max_yaw[boneIndex];
			m_currentPoseStats.avg_avg_yaws += m_currentPoseStats.avg_yaw[boneIndex];
		}

		m_currentPoseStats.avg_max_yaws /= NUM_BONES_FOR_YAW_STAT;
		m_currentPoseStats.avg_avg_yaws /= NUM_BONES_FOR_YAW_STAT;

		m_samplePoseHeadIndex = 0;
	}
};



#endif

