#include "PFNNBaseCode.h"


#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>

/* Some private Helper Functions */
glm::vec3 mix_directions(glm::vec3 x, glm::vec3 y, float a) {
	glm::quat x_q = glm::angleAxis(atan2f(x.x, x.z), glm::vec3(0, 1, 0));
	glm::quat y_q = glm::angleAxis(atan2f(y.x, y.z), glm::vec3(0, 1, 0));
	glm::quat z_q = glm::slerp(x_q, y_q, a);
	return z_q * glm::vec3(0, 0, 1);
}

glm::mat4 mix_transforms(glm::mat4 x, glm::mat4 y, float a) {
	glm::mat4 out = glm::mat4(glm::slerp(glm::quat(x), glm::quat(y), a));
	out[3] = mix(x[3], y[3], a);
	return out;
}

glm::quat quat_exp(glm::vec3 l) {
	float w = glm::length(l);
	glm::quat q = w < 0.01 ? glm::quat(1, 0, 0, 0) : glm::quat(
		cosf(w),
		l.x * (sinf(w) / w),
		l.y * (sinf(w) / w),
		l.z * (sinf(w) / w));
	return q / sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

glm::vec2 segment_nearest(glm::vec2 v, glm::vec2 w, glm::vec2 p) {
	float l2 = glm::dot(v - w, v - w);
	if (l2 == 0.0) return v;
	float t = glm::clamp(glm::dot(p - v, w - v) / l2, 0.0f, 1.0f);
	return v + t * (w - v);
}


//////////////////////////////////////////// ENDED DATA DEFINITIONS

void Character::setDesiredSpeed(const float desiredSpeed_cm)
{
	const float desiredSpeed_m = desiredSpeed_cm * UNITCONV_CM_TO_M; // to meters

	m_INPUT_SPEED = 0.0f;
	float XPower_desiredSpeed = 1.0f;
	for (int i = 0; i < speedConverter_degree; i++)
	{
		m_INPUT_SPEED += XPower_desiredSpeed * speedConverter_coefficients[i];
		XPower_desiredSpeed *= desiredSpeed_m;
	}

	m_INPUT_SPEED = std::max(0.0f, m_INPUT_SPEED) * UNITCONV_M_TO_CM;
}


void PFNNCharacterBase::init(const std::string& strAssetsDir)
{
	character = new Character();
	trajectory = new Trajectory();
	options = new Options();
	pfnn = new PFNN(PFNN::MODE_CONSTANT);
	pfnn->load(strAssetsDir);

	character->load(
		(strAssetsDir + "Parameters/character_vertices.bin").c_str(),
		(strAssetsDir + "Parameters/character_triangles.bin").c_str(),
		(strAssetsDir + "Parameters/character_parents.bin").c_str(),
		(strAssetsDir + "Parameters/character_xforms.bin").c_str(),
		(strAssetsDir + "Parameters/SpeedConvertorData/speedFactors.bin").c_str());

	resetPosAndOrientation(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f));
}


void PFNNCharacterBase::resetPosAndOrientation(glm::vec2 position, glm::vec2 orientation)
{
	ArrayXf Yp = pfnn->Ymean;
	glm::vec3 root_position = glm::vec3(position.x, 0.0f /*heightmap->sample(position)*/, position.y);

    const float orientationAngle2D = atan2(orientation.y, orientation.x) * (180.0 / M_PI);
	glm::mat3 root_rotation = glm::mat3(glm::rotate(orientationAngle2D, glm::vec3(0.0f, 1.0f, 0.0)));

    //std::cout<<root_rotation;

    glm::vec3 forward = root_rotation[2];

	for (int i = 0; i < Trajectory::LENGTH; i++) {
		trajectory->positions[i] = root_position;
		trajectory->rotations[i] = root_rotation;
		trajectory->directions[i] = forward; //glm::vec3(0, 0, 1);
		trajectory->heights[i] = root_position.y;
		trajectory->gait_stand[i] = 1.0;
		trajectory->gait_walk[i] = 0.0;
		trajectory->gait_jog[i] = 0.0;
		trajectory->gait_crouch[i] = 0.0;
		trajectory->gait_jump[i] = 0.0;
		trajectory->gait_bump[i] = 0.0;
	}

	float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::min();
	float minY = minX, maxY = maxX;
	float minZ = minX, maxZ = maxX;

	for (int i = 0; i < Character::JOINT_NUM; i++) {

		int opos = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 0);
		int ovel = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 1);
		int orot = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 2);

		glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
		glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
		glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));

		character->joint_positions[i] = pos;
		character->joint_velocities[i] = vel;
		character->joint_rotations[i] = rot;

		minX = std::min(minX, pos.x);
		minY = std::min(minY, pos.y);
		minZ = std::min(minZ, pos.z);
		maxX = std::max(maxX, pos.x);
		maxY = std::max(maxY, pos.y);
		maxZ = std::max(maxZ, pos.z);

		character->joint_global_anim_xform[i] = glm::transpose(glm::mat4(
			rot[0][0], rot[1][0], rot[2][0], pos[0],
			rot[0][1], rot[1][1], rot[2][1], pos[1],
			rot[0][2], rot[1][2], rot[2][2], pos[2],
			0, 0, 0, 1));
	}

	character->bbox_xmin = minX;
	character->bbox_xmax = maxX;
	character->bbox_ymin = minY;
	character->bbox_ymax = maxY;
	character->bbox_zmin = minZ;
	character->bbox_zmax = maxZ;

	/* Convert to local space ... yes I know this is inefficient. */
	for (int i = 0; i < Character::JOINT_NUM; i++)
	{
		if (i == 0) {
			character->joint_anim_xform[i] = character->joint_global_anim_xform[i];
		}
		else {
			character->joint_anim_xform[i] = glm::inverse(character->joint_global_anim_xform[character->joint_parents[i]]) * character->joint_global_anim_xform[i];
		} 

		character->joint_velocities[i] = glm::vec3(0.0f);
	}
	character->forward_kinematics();

	character->phase = 0.0;
	/*
	ik->position[IK::HL] = glm::vec3(0, 0, 0); ik->lock[IK::HL] = 0; ik->height[IK::HL] = root_position.y;
	ik->position[IK::HR] = glm::vec3(0, 0, 0); ik->lock[IK::HR] = 0; ik->height[IK::HR] = root_position.y;
	ik->position[IK::TL] = glm::vec3(0, 0, 0); ik->lock[IK::TL] = 0; ik->height[IK::TL] = root_position.y;
	ik->position[IK::TR] = glm::vec3(0, 0, 0); ik->lock[IK::TR] = 0; ik->height[IK::TR] = root_position.y;
	*/

	//float bboxX, bboxY, bboxZ;
	//character->getBboxSize(bboxX, bboxY, bboxZ);
}

void PFNNCharacterBase::updateAnim(float deltaTime)
{
#if 0
	/* Update Camera */
	int x_move = SDL_JoystickGetAxis(stick, GAMEPAD_STICK_R_HORIZONTAL);
	int y_move = SDL_JoystickGetAxis(stick, GAMEPAD_STICK_R_VERTICAL);

	if (abs(x_move) + abs(y_move) < 10000) { x_move = 0; y_move = 0; };


	if (options->invert_y) { y_move = -y_move; }

	camera->pitch = glm::clamp(camera->pitch + (y_move / 32768.0) * 0.03, M_PI / 16, 2 * M_PI / 5);
	camera->yaw = camera->yaw + (x_move / 32768.0) * 0.03;


	float zoom_i = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_L) * 20.0;
	float zoom_o = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_R) * 20.0;

	if (zoom_i > 1e-5) { camera->distance = glm::clamp(camera->distance + zoom_i, 10.0f, 10000.0f); }
	if (zoom_o > 1e-5) { camera->distance = glm::clamp(camera->distance - zoom_o, 10.0f, 10000.0f); }

	/* Update Target Direction / Velocity */

	int x_vel = -SDL_JoystickGetAxis(stick, GAMEPAD_STICK_L_HORIZONTAL);
	int y_vel = -SDL_JoystickGetAxis(stick, GAMEPAD_STICK_L_VERTICAL);
	if (abs(x_vel) + abs(y_vel) < 10000) { x_vel = 0; y_vel = 0; };

	glm::vec3 trajectory_target_direction_new = glm::normalize(glm::vec3(camera->direction().x, 0.0, camera->direction().z));
	glm::mat3 trajectory_target_rotation = glm::mat3(glm::rotate(atan2f(
		trajectory_target_direction_new.x,
		trajectory_target_direction_new.z), glm::vec3(0, 1, 0)));
#endif
	glm::mat3 trajectory_target_rotation = glm::mat3(1.0f);

	float target_vel_speed = character->m_INPUT_SPEED * deltaTime; // 2.5 + 2.5 * ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_R) / 32768.0) + 1.0);


	glm::vec3 rootPosition = trajectory->positions[Trajectory::LENGTH / 2];
	glm::vec3 dirRootToTarget = g_nextTargetToGo - rootPosition;
	float distToTarget = glm::length(dirRootToTarget);
	glm::vec3 dirRootToTargetNorm = distToTarget == 0.0f ? dirRootToTarget : glm::normalize(dirRootToTarget);



	glm::vec3 trajectory_target_velocity_new = target_vel_speed * dirRootToTargetNorm; //(trajectory_target_rotation * dirRootToTargetNorm);//*glm::vec3(x_vel / 32768.0, 0, y_vel / 32768.0));
	trajectory->target_vel = glm::mix(trajectory->target_vel, trajectory_target_velocity_new, options->extra_velocity_smooth);

#if 1
	// Prevent overshooting
	if (glm::length(trajectory->target_vel) > distToTarget)
	{
		trajectory->target_vel = glm::normalize(trajectory->target_vel) * distToTarget;
	}
#endif


	character->strafe_target = 0.0f; // USE_STRAFE ? ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_L) / 32768.0) + 1.0) / 2.0 : 0.0f;
	character->strafe_amount = glm::mix(character->strafe_amount, character->strafe_target, options->extra_strafe_smooth);

	glm::vec3 trajectory_target_velocity_dir = glm::length(trajectory->target_vel) < 1e-05 ? trajectory->target_dir : glm::normalize(trajectory->target_vel);
	glm::vec3 trajectory_target_direction_new = trajectory_target_velocity_dir; // mix_directions(trajectory_target_velocity_dir, trajectory_target_direction_new, character->strafe_amount);
	trajectory->target_dir = mix_directions(trajectory->target_dir, trajectory_target_direction_new, options->extra_direction_smooth);

	character->crouched_amount = glm::mix(character->crouched_amount, character->crouched_target, options->extra_crouched_smooth);

	/* Update Gait */

	if (glm::length(trajectory->target_vel) < m_targetReachedThreshold) 
	{
		// REset target velocity to zero since we want to stand
		trajectory->target_vel = glm::vec3(0.0f);

		float stand_amount = 1.0f - glm::clamp(glm::length(trajectory->target_vel) / 0.1f, 0.0f, 1.0f);
		trajectory->gait_stand[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_stand[Trajectory::LENGTH / 2], stand_amount, options->extra_gait_smooth);
		trajectory->gait_walk[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_walk[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_jog[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jog[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_crouch[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_jump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_bump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_bump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
	}
	else if (character->crouched_amount > 0.1) {
		trajectory->gait_stand[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_stand[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_walk[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_walk[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_jog[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jog[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_crouch[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH / 2], character->crouched_amount, options->extra_gait_smooth);
		trajectory->gait_jump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_bump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_bump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
	}
	else if (true) //(SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_R) / 32768.0) + 1.0) {
	{
		// COmpute balance between jog and walk
		const float JOG_START_SPEED = 9500;
		const float JOG_END_SPEED = 20000;
		const float currentInputSpeed = character->m_INPUT_SPEED;
		const float t = std::min(1.0f, std::max(0.0f, (currentInputSpeed - JOG_START_SPEED) / (JOG_END_SPEED - JOG_START_SPEED))); // Clamp [0,1]
		const float walk_f = 1 - t;
		const float jog_f = t;

		trajectory->gait_stand[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_stand[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_walk[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_walk[Trajectory::LENGTH / 2], walk_f, options->extra_gait_smooth);
		trajectory->gait_jog[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jog[Trajectory::LENGTH / 2], jog_f, options->extra_gait_smooth);
		trajectory->gait_crouch[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_jump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_bump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_bump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
	}
	else
	{
		trajectory->gait_stand[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_stand[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_walk[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_walk[Trajectory::LENGTH / 2], 1.0f, options->extra_gait_smooth);
		trajectory->gait_jog[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jog[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_crouch[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_jump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_jump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
		trajectory->gait_bump[Trajectory::LENGTH / 2] = glm::mix(trajectory->gait_bump[Trajectory::LENGTH / 2], 0.0f, options->extra_gait_smooth);
	}

	/* Predict Future Trajectory */

	glm::vec3 trajectory_positions_blend[Trajectory::LENGTH];
	trajectory_positions_blend[Trajectory::LENGTH / 2] = trajectory->positions[Trajectory::LENGTH / 2];


	for (int i = Trajectory::LENGTH / 2 + 1; i < Trajectory::LENGTH; i++) 
	{
#ifdef USE_DIRECT_TARGET_MOTION
		const glm::vec3 fixedSizeBetweenPoints = trajectory->target_vel * (1.0f / (Trajectory::LENGTH / 2));
		trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + fixedSizeBetweenPoints;
		trajectory->directions[i] = trajectory->target_dir;
#else
		float bias_pos = character->responsive ? glm::mix(2.0f, 2.0f, character->strafe_amount) : glm::mix(0.5f, 1.0f, character->strafe_amount);
		float bias_dir = character->responsive ? glm::mix(5.0f, 3.0f, character->strafe_amount) : glm::mix(2.0f, 0.5f, character->strafe_amount);



		float scale_pos = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH / 2) / (Trajectory::LENGTH / 2)), bias_pos));
		float scale_dir = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH / 2) / (Trajectory::LENGTH / 2)), bias_dir));

		trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + glm::mix(
			trajectory->positions[i] - trajectory->positions[i - 1],
			trajectory->target_vel,
			scale_pos);


		trajectory->directions[i] = mix_directions(trajectory->directions[i], trajectory->target_dir, scale_dir);
#endif

		/* Collide with walls */
#if 0
		for (int j = 0; j < areas->num_walls(); j++) {
			glm::vec2 trjpoint = glm::vec2(trajectory_positions_blend[i].x, trajectory_positions_blend[i].z);
			if (glm::length(trjpoint - ((areas->wall_start[j] + areas->wall_stop[j]) / 2.0f)) >
				glm::length(areas->wall_start[j] - areas->wall_stop[j])) {
				continue;
			}
			glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
			float segdist = glm::length(segpoint - trjpoint);
			if (segdist < areas->wall_width[j] + 100.0) {
				glm::vec2 prjpoint0 = (areas->wall_width[j] + 0.0f) * glm::normalize(trjpoint - segpoint) + segpoint;
				glm::vec2 prjpoint1 = (areas->wall_width[j] + 100.0f) * glm::normalize(trjpoint - segpoint) + segpoint;
				glm::vec2 prjpoint = glm::mix(prjpoint0, prjpoint1, glm::clamp((segdist - areas->wall_width[j]) / 100.0f, 0.0f, 1.0f));
				trajectory_positions_blend[i].x = prjpoint.x;
				trajectory_positions_blend[i].z = prjpoint.y;
			}
		}
#endif

		trajectory->heights[i] = trajectory->heights[Trajectory::LENGTH / 2];

		trajectory->gait_stand[i] = trajectory->gait_stand[Trajectory::LENGTH / 2];
		trajectory->gait_walk[i] = trajectory->gait_walk[Trajectory::LENGTH / 2];
		trajectory->gait_jog[i] = trajectory->gait_jog[Trajectory::LENGTH / 2];
		trajectory->gait_crouch[i] = trajectory->gait_crouch[Trajectory::LENGTH / 2];
		trajectory->gait_jump[i] = trajectory->gait_jump[Trajectory::LENGTH / 2];
		trajectory->gait_bump[i] = trajectory->gait_bump[Trajectory::LENGTH / 2];
	}

	for (int i = Trajectory::LENGTH / 2 + 1; i < Trajectory::LENGTH; i++) {
		trajectory->positions[i] = trajectory_positions_blend[i];
	}


	const float sanityLenTotal = glm::distance(trajectory->positions[Trajectory::LENGTH / 2], trajectory->positions[Trajectory::LENGTH - 1]);


	/* Jumps */
#if 0
	for (int i = Trajectory::LENGTH / 2; i < Trajectory::LENGTH; i++) {
		trajectory->gait_jump[i] = 0.0;
		for (int j = 0; j < areas->num_jumps(); j++) {
			float dist = glm::length(trajectory->positions[i] - areas->jump_pos[j]);
			trajectory->gait_jump[i] = std::max(trajectory->gait_jump[i],
				1.0f - glm::clamp((dist - areas->jump_size[j]) / areas->jump_falloff[j], 0.0f, 1.0f));
		}
	}

	/* Crouch Area */
	for (int i = Trajectory::LENGTH / 2; i < Trajectory::LENGTH; i++) {
		for (int j = 0; j < areas->num_crouches(); j++) {
			float dist_x = abs(trajectory->positions[i].x - areas->crouch_pos[j].x);
			float dist_z = abs(trajectory->positions[i].z - areas->crouch_pos[j].z);
			float height = (sinf(trajectory->positions[i].x / Areas::CROUCH_WAVE) + 1.0) / 2.0;
			trajectory->gait_crouch[i] = glm::mix(1.0f - height, trajectory->gait_crouch[i],
				glm::clamp(
				((dist_x - (areas->crouch_size[j].x / 2)) +
					(dist_z - (areas->crouch_size[j].y / 2))) / 100.0f, 0.0f, 1.0f));
		}
	}

	/* Walls */
	for (int i = 0; i < Trajectory::LENGTH; i++) {
		trajectory->gait_bump[i] = 0.0;
		for (int j = 0; j < areas->num_walls(); j++) {
			glm::vec2 trjpoint = glm::vec2(trajectory->positions[i].x, trajectory->positions[i].z);
			glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
			float segdist = glm::length(segpoint - trjpoint);
			trajectory->gait_bump[i] = glm::max(trajectory->gait_bump[i], 1.0f - glm::clamp((segdist - areas->wall_width[j]) / 10.0f, 0.0f, 1.0f));
		}
	}
#endif

	/* Trajectory Rotation */
	for (int i = 0; i < Trajectory::LENGTH; i++) {
		trajectory->rotations[i] = glm::mat3(glm::rotate(atan2f(
			trajectory->directions[i].x,
			trajectory->directions[i].z), glm::vec3(0, 1, 0)));
	}

	// TODO HERE ALSO
	/* Trajectory Heights */
	for (int i = Trajectory::LENGTH / 2; i < Trajectory::LENGTH; i++) {
		trajectory->positions[i].y = 0.0f; // heightmap->sample(glm::vec2(trajectory->positions[i].x, trajectory->positions[i].z));
	}

	trajectory->heights[Trajectory::LENGTH / 2] = 0.0;
	for (int i = 0; i < Trajectory::LENGTH; i += 10) {
		trajectory->heights[Trajectory::LENGTH / 2] += (trajectory->positions[i].y / ((Trajectory::LENGTH) / 10));
	}

	glm::vec3 root_position = glm::vec3(
		trajectory->positions[Trajectory::LENGTH / 2].x,
		trajectory->heights[Trajectory::LENGTH / 2],
		trajectory->positions[Trajectory::LENGTH / 2].z);

	glm::mat3 root_rotation = trajectory->rotations[Trajectory::LENGTH / 2];

	/* Input Trajectory Positions / Directions */
	for (int i = 0; i < Trajectory::LENGTH; i += 10) {
		int w = (Trajectory::LENGTH) / 10;
		glm::vec3 pos = glm::inverse(root_rotation) * (trajectory->positions[i] - root_position);
		glm::vec3 dir = glm::inverse(root_rotation) * trajectory->directions[i];
		pfnn->Xp((w * 0) + i / 10) = pos.x; pfnn->Xp((w * 1) + i / 10) = pos.z;
		pfnn->Xp((w * 2) + i / 10) = dir.x; pfnn->Xp((w * 3) + i / 10) = dir.z;
	}

	/* Input Trajectory Gaits */
	for (int i = 0; i < Trajectory::LENGTH; i += 10) {
		int w = (Trajectory::LENGTH) / 10;
		pfnn->Xp((w * 4) + i / 10) = trajectory->gait_stand[i];
		pfnn->Xp((w * 5) + i / 10) = trajectory->gait_walk[i];
		pfnn->Xp((w * 6) + i / 10) = trajectory->gait_jog[i];
		pfnn->Xp((w * 7) + i / 10) = trajectory->gait_crouch[i];
		pfnn->Xp((w * 8) + i / 10) = trajectory->gait_jump[i];
		pfnn->Xp((w * 9) + i / 10) = 0.0; // Unused.
	}

	/* Input Joint Previous Positions / Velocities / Rotations */
	glm::vec3 prev_root_position = glm::vec3(
		trajectory->positions[Trajectory::LENGTH / 2 - 1].x,
		trajectory->heights[Trajectory::LENGTH / 2 - 1],
		trajectory->positions[Trajectory::LENGTH / 2 - 1].z);

	glm::mat3 prev_root_rotation = trajectory->rotations[Trajectory::LENGTH / 2 - 1];

	for (int i = 0; i < Character::JOINT_NUM; i++) {
		int o = (((Trajectory::LENGTH) / 10) * 10);
		glm::vec3 pos = glm::inverse(prev_root_rotation) * (character->joint_positions[i] - prev_root_position);
		glm::vec3 prv = glm::inverse(prev_root_rotation) *  character->joint_velocities[i];
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 0) + i * 3 + 0) = pos.x;
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 0) + i * 3 + 1) = pos.y;
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 0) + i * 3 + 2) = pos.z;
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 1) + i * 3 + 0) = prv.x;
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 1) + i * 3 + 1) = prv.y;
		pfnn->Xp(o + (Character::JOINT_NUM * 3 * 1) + i * 3 + 2) = prv.z;
	}

	/* Input Trajectory Heights */
	for (int i = 0; i < Trajectory::LENGTH; i += 10) {
		int o = (((Trajectory::LENGTH) / 10) * 10) + Character::JOINT_NUM * 3 * 2;
		int w = (Trajectory::LENGTH) / 10;
		glm::vec3 position_r = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(trajectory->width, 0, 0));
		glm::vec3 position_l = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(-trajectory->width, 0, 0));
		pfnn->Xp(o + (w * 0) + (i / 10)) = 0.0f; // heightmap->sample(glm::vec2(position_r.x, position_r.z)) - root_position.y;
		pfnn->Xp(o + (w * 1) + (i / 10)) = trajectory->positions[i].y - root_position.y;
		pfnn->Xp(o + (w * 2) + (i / 10)) = 0.0f; // heightmap->sample(glm::vec2(position_l.x, position_l.z)) - root_position.y;
	}

	/* Perform Regression */

	//clock_t time_start = clock();

	pfnn->predict(character->phase);

	//clock_t time_end = clock();

	/* Timing */
#if 0
	enum { TIME_MSAMPLES = 500 };
	static int time_nsamples = 0;
	static float time_samples[TIME_MSAMPLES];

	time_samples[time_nsamples] = (float)(time_end - time_start) / CLOCKS_PER_SEC;
	time_nsamples++;
	if (time_nsamples == TIME_MSAMPLES) {
		float time_avg = 0.0;
		for (int i = 0; i < TIME_MSAMPLES; i++) {
			time_avg += (time_samples[i] / TIME_MSAMPLES);
		}
		printf("PFNN: %0.5f ms\n", time_avg * 1000);
		time_nsamples = 0;
	}
#endif 

	/* Build Local Transforms */

	for (int i = 0; i < Character::JOINT_NUM; i++) {
		int opos = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 0);
		int ovel = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 1);
		int orot = 8 + (((Trajectory::LENGTH / 2) / 10) * 4) + (Character::JOINT_NUM * 3 * 2);

		glm::mat3(glm::rotate(atan2f(
			trajectory->directions[i].x,
			trajectory->directions[i].z), glm::vec3(0, 1, 0)));

		glm::vec3 pos = (root_rotation * glm::vec3(pfnn->Yp(opos + i * 3 + 0), pfnn->Yp(opos + i * 3 + 1), pfnn->Yp(opos + i * 3 + 2))) + root_position;
		glm::vec3 vel = (root_rotation * glm::vec3(pfnn->Yp(ovel + i * 3 + 0), pfnn->Yp(ovel + i * 3 + 1), pfnn->Yp(ovel + i * 3 + 2)));

		const glm::quat outRotationQuat = quat_exp(glm::vec3(pfnn->Yp(orot + i * 3 + 0), pfnn->Yp(orot + i * 3 + 1), pfnn->Yp(orot + i * 3 + 2)));

		glm::vec3 eulerAngles = glm::eulerAngles(outRotationQuat);
		glm::mat3 rot = (root_rotation * glm::toMat3(outRotationQuat));

		
		/*
		** Blending Between the predicted positions and
		** the previous positions plus the velocities
		** smooths out the motion a bit in the case
		** where the two disagree with each other.
		*/

		character->joint_positions[i] = glm::mix(character->joint_positions[i] + vel, pos, options->extra_joint_smooth);
		character->joint_velocities[i] = vel;
		character->joint_rotations[i] = rot;

		character->joint_global_anim_xform[i] = glm::transpose(glm::mat4(
			rot[0][0], rot[1][0], rot[2][0], pos[0],
			rot[0][1], rot[1][1], rot[2][1], pos[1],
			rot[0][2], rot[1][2], rot[2][2], pos[2],
			0, 0, 0, 1));
	}

	/* Convert to local space ... yes I know this is inefficient. */

	for (int i = 0; i < Character::JOINT_NUM; i++) {
		if (i == 0) {
			character->joint_anim_xform[i] = character->joint_global_anim_xform[i];
		}
		else {
			character->joint_anim_xform[i] = glm::inverse(character->joint_global_anim_xform[character->joint_parents[i]]) * character->joint_global_anim_xform[i];
		}
	}

	character->forward_kinematics();

#if 0
	/* Perform IK (enter this block at your own risk...) */
	if (options->enable_ik) {

		/* Get Weights */

		glm::vec4 ik_weight = glm::vec4(pfnn->Yp(4 + 0), pfnn->Yp(4 + 1), pfnn->Yp(4 + 2), pfnn->Yp(4 + 3));

		glm::vec3 key_hl = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_L][3]);
		glm::vec3 key_tl = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_L][3]);
		glm::vec3 key_hr = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_R][3]);
		glm::vec3 key_tr = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_R][3]);

		key_hl = glm::mix(key_hl, ik->position[IK::HL], ik->lock[IK::HL]);
		key_tl = glm::mix(key_tl, ik->position[IK::TL], ik->lock[IK::TL]);
		key_hr = glm::mix(key_hr, ik->position[IK::HR], ik->lock[IK::HR]);
		key_tr = glm::mix(key_tr, ik->position[IK::TR], ik->lock[IK::TR]);

		ik->height[IK::HL] = glm::mix(ik->height[IK::HL], heightmap->sample(glm::vec2(key_hl.x, key_hl.z)) + ik->heel_height, ik->smoothness);
		ik->height[IK::TL] = glm::mix(ik->height[IK::TL], heightmap->sample(glm::vec2(key_tl.x, key_tl.z)) + ik->toe_height, ik->smoothness);
		ik->height[IK::HR] = glm::mix(ik->height[IK::HR], heightmap->sample(glm::vec2(key_hr.x, key_hr.z)) + ik->heel_height, ik->smoothness);
		ik->height[IK::TR] = glm::mix(ik->height[IK::TR], heightmap->sample(glm::vec2(key_tr.x, key_tr.z)) + ik->toe_height, ik->smoothness);

		key_hl.y = glm::max(key_hl.y, ik->height[IK::HL]);
		key_tl.y = glm::max(key_tl.y, ik->height[IK::TL]);
		key_hr.y = glm::max(key_hr.y, ik->height[IK::HR]);
		key_tr.y = glm::max(key_tr.y, ik->height[IK::TR]);

		/* Rotate Hip / Knee */

		{
			glm::vec3 hip_l = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HIP_L][3]);
			glm::vec3 knee_l = glm::vec3(character->joint_global_anim_xform[Character::JOINT_KNEE_L][3]);
			glm::vec3 heel_l = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_L][3]);

			glm::vec3 hip_r = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HIP_R][3]);
			glm::vec3 knee_r = glm::vec3(character->joint_global_anim_xform[Character::JOINT_KNEE_R][3]);
			glm::vec3 heel_r = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_R][3]);

			ik->two_joint(hip_l, knee_l, heel_l, key_hl, 1.0,
				character->joint_global_anim_xform[Character::JOINT_ROOT_L],
				character->joint_global_anim_xform[Character::JOINT_HIP_L],
				character->joint_global_anim_xform[Character::JOINT_HIP_L],
				character->joint_global_anim_xform[Character::JOINT_KNEE_L],
				character->joint_anim_xform[Character::JOINT_HIP_L],
				character->joint_anim_xform[Character::JOINT_KNEE_L]);

			ik->two_joint(hip_r, knee_r, heel_r, key_hr, 1.0,
				character->joint_global_anim_xform[Character::JOINT_ROOT_R],
				character->joint_global_anim_xform[Character::JOINT_HIP_R],
				character->joint_global_anim_xform[Character::JOINT_HIP_R],
				character->joint_global_anim_xform[Character::JOINT_KNEE_R],
				character->joint_anim_xform[Character::JOINT_HIP_R],
				character->joint_anim_xform[Character::JOINT_KNEE_R]);

			character->forward_kinematics();
		}

		/* Rotate Heel */

		{
			const float heel_max_bend_s = 4;
			const float heel_max_bend_u = 4;
			const float heel_max_bend_d = 4;

			glm::vec4 ik_toe_pos_blend = glm::clamp(ik_weight * 2.5f, 0.0f, 1.0f);

			glm::vec3 heel_l = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_L][3]);
			glm::vec4 side_h0_l = character->joint_global_anim_xform[Character::JOINT_HEEL_L] * glm::vec4(10, 0, 0, 1);
			glm::vec4 side_h1_l = character->joint_global_anim_xform[Character::JOINT_HEEL_L] * glm::vec4(-10, 0, 0, 1);
			glm::vec3 side0_l = glm::vec3(side_h0_l) / side_h0_l.w;
			glm::vec3 side1_l = glm::vec3(side_h1_l) / side_h1_l.w;
			glm::vec3 floor_l = key_tl;

			side0_l.y = glm::clamp(heightmap->sample(glm::vec2(side0_l.x, side0_l.z)) + ik->toe_height, heel_l.y - heel_max_bend_s, heel_l.y + heel_max_bend_s);
			side1_l.y = glm::clamp(heightmap->sample(glm::vec2(side1_l.x, side1_l.z)) + ik->toe_height, heel_l.y - heel_max_bend_s, heel_l.y + heel_max_bend_s);
			floor_l.y = glm::clamp(floor_l.y, heel_l.y - heel_max_bend_d, heel_l.y + heel_max_bend_u);

			glm::vec3 targ_z_l = glm::normalize(floor_l - heel_l);
			glm::vec3 targ_x_l = glm::normalize(side0_l - side1_l);
			glm::vec3 targ_y_l = glm::normalize(glm::cross(targ_x_l, targ_z_l));
			targ_x_l = glm::cross(targ_z_l, targ_y_l);

			character->joint_anim_xform[Character::JOINT_HEEL_L] = mix_transforms(
				character->joint_anim_xform[Character::JOINT_HEEL_L],
				glm::inverse(character->joint_global_anim_xform[Character::JOINT_KNEE_L]) * glm::mat4(
					glm::vec4(targ_x_l, 0),
					glm::vec4(-targ_y_l, 0),
					glm::vec4(targ_z_l, 0),
					glm::vec4(heel_l, 1)), ik_toe_pos_blend.y);

			glm::vec3 heel_r = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_R][3]);
			glm::vec4 side_h0_r = character->joint_global_anim_xform[Character::JOINT_HEEL_R] * glm::vec4(10, 0, 0, 1);
			glm::vec4 side_h1_r = character->joint_global_anim_xform[Character::JOINT_HEEL_R] * glm::vec4(-10, 0, 0, 1);
			glm::vec3 side0_r = glm::vec3(side_h0_r) / side_h0_r.w;
			glm::vec3 side1_r = glm::vec3(side_h1_r) / side_h1_r.w;
			glm::vec3 floor_r = key_tr;

			side0_r.y = glm::clamp(heightmap->sample(glm::vec2(side0_r.x, side0_r.z)) + ik->toe_height, heel_r.y - heel_max_bend_s, heel_r.y + heel_max_bend_s);
			side1_r.y = glm::clamp(heightmap->sample(glm::vec2(side1_r.x, side1_r.z)) + ik->toe_height, heel_r.y - heel_max_bend_s, heel_r.y + heel_max_bend_s);
			floor_r.y = glm::clamp(floor_r.y, heel_r.y - heel_max_bend_d, heel_r.y + heel_max_bend_u);

			glm::vec3 targ_z_r = glm::normalize(floor_r - heel_r);
			glm::vec3 targ_x_r = glm::normalize(side0_r - side1_r);
			glm::vec3 targ_y_r = glm::normalize(glm::cross(targ_z_r, targ_x_r));
			targ_x_r = glm::cross(targ_z_r, targ_y_r);

			character->joint_anim_xform[Character::JOINT_HEEL_R] = mix_transforms(
				character->joint_anim_xform[Character::JOINT_HEEL_R],
				glm::inverse(character->joint_global_anim_xform[Character::JOINT_KNEE_R]) * glm::mat4(
					glm::vec4(-targ_x_r, 0),
					glm::vec4(targ_y_r, 0),
					glm::vec4(targ_z_r, 0),
					glm::vec4(heel_r, 1)), ik_toe_pos_blend.w);

			character->forward_kinematics();
		}

		/* Rotate Toe */

		{
			const float toe_max_bend_d = 0;
			const float toe_max_bend_u = 10;

			glm::vec4 ik_toe_rot_blend = glm::clamp(ik_weight * 2.5f, 0.0f, 1.0f);

			glm::vec3 toe_l = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_L][3]);
			glm::vec4 fwrd_h_l = character->joint_global_anim_xform[Character::JOINT_TOE_L] * glm::vec4(0, 0, 10, 1);
			glm::vec4 side_h0_l = character->joint_global_anim_xform[Character::JOINT_TOE_L] * glm::vec4(10, 0, 0, 1);
			glm::vec4 side_h1_l = character->joint_global_anim_xform[Character::JOINT_TOE_L] * glm::vec4(-10, 0, 0, 1);
			glm::vec3 fwrd_l = glm::vec3(fwrd_h_l) / fwrd_h_l.w;
			glm::vec3 side0_l = glm::vec3(side_h0_l) / side_h0_l.w;
			glm::vec3 side1_l = glm::vec3(side_h1_l) / side_h1_l.w;

			fwrd_l.y = glm::clamp(heightmap->sample(glm::vec2(fwrd_l.x, fwrd_l.z)) + ik->toe_height, toe_l.y - toe_max_bend_d, toe_l.y + toe_max_bend_u);
			side0_l.y = glm::clamp(heightmap->sample(glm::vec2(side0_l.x, side0_l.z)) + ik->toe_height, toe_l.y - toe_max_bend_d, toe_l.y + toe_max_bend_u);
			side1_l.y = glm::clamp(heightmap->sample(glm::vec2(side0_l.x, side1_l.z)) + ik->toe_height, toe_l.y - toe_max_bend_d, toe_l.y + toe_max_bend_u);

			glm::vec3 side_l = glm::normalize(side0_l - side1_l);
			fwrd_l = glm::normalize(fwrd_l - toe_l);
			glm::vec3 upwr_l = glm::normalize(glm::cross(side_l, fwrd_l));
			side_l = glm::cross(fwrd_l, upwr_l);

			character->joint_anim_xform[Character::JOINT_TOE_L] = mix_transforms(
				character->joint_anim_xform[Character::JOINT_TOE_L],
				glm::inverse(character->joint_global_anim_xform[Character::JOINT_HEEL_L]) * glm::mat4(
					glm::vec4(side_l, 0),
					glm::vec4(-upwr_l, 0),
					glm::vec4(fwrd_l, 0),
					glm::vec4(toe_l, 1)), ik_toe_rot_blend.y);

			glm::vec3 toe_r = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_R][3]);
			glm::vec4 fwrd_h_r = character->joint_global_anim_xform[Character::JOINT_TOE_R] * glm::vec4(0, 0, 10, 1);
			glm::vec4 side_h0_r = character->joint_global_anim_xform[Character::JOINT_TOE_R] * glm::vec4(10, 0, 0, 1);
			glm::vec4 side_h1_r = character->joint_global_anim_xform[Character::JOINT_TOE_R] * glm::vec4(-10, 0, 0, 1);
			glm::vec3 fwrd_r = glm::vec3(fwrd_h_r) / fwrd_h_r.w;
			glm::vec3 side0_r = glm::vec3(side_h0_r) / side_h0_r.w;
			glm::vec3 side1_r = glm::vec3(side_h1_r) / side_h1_r.w;

			fwrd_r.y = glm::clamp(heightmap->sample(glm::vec2(fwrd_r.x, fwrd_r.z)) + ik->toe_height, toe_r.y - toe_max_bend_d, toe_r.y + toe_max_bend_u);
			side0_r.y = glm::clamp(heightmap->sample(glm::vec2(side0_r.x, side0_r.z)) + ik->toe_height, toe_r.y - toe_max_bend_d, toe_r.y + toe_max_bend_u);
			side1_r.y = glm::clamp(heightmap->sample(glm::vec2(side1_r.x, side1_r.z)) + ik->toe_height, toe_r.y - toe_max_bend_d, toe_r.y + toe_max_bend_u);

			glm::vec3 side_r = glm::normalize(side0_r - side1_r);
			fwrd_r = glm::normalize(fwrd_r - toe_r);
			glm::vec3 upwr_r = glm::normalize(glm::cross(side_r, fwrd_r));
			side_r = glm::cross(fwrd_r, upwr_r);

			character->joint_anim_xform[Character::JOINT_TOE_R] = mix_transforms(
				character->joint_anim_xform[Character::JOINT_TOE_R],
				glm::inverse(character->joint_global_anim_xform[Character::JOINT_HEEL_R]) * glm::mat4(
					glm::vec4(side_r, 0),
					glm::vec4(-upwr_r, 0),
					glm::vec4(fwrd_r, 0),
					glm::vec4(toe_r, 1)), ik_toe_rot_blend.w);

			character->forward_kinematics();
		}

		/* Update Locks */

		if ((ik->lock[IK::HL] == 0.0) && (ik_weight.y >= ik->threshold)) {
			ik->lock[IK::HL] = 1.0; ik->position[IK::HL] = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_L][3]);
			ik->lock[IK::TL] = 1.0; ik->position[IK::TL] = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_L][3]);
		}

		if ((ik->lock[IK::HR] == 0.0) && (ik_weight.w >= ik->threshold)) {
			ik->lock[IK::HR] = 1.0; ik->position[IK::HR] = glm::vec3(character->joint_global_anim_xform[Character::JOINT_HEEL_R][3]);
			ik->lock[IK::TR] = 1.0; ik->position[IK::TR] = glm::vec3(character->joint_global_anim_xform[Character::JOINT_TOE_R][3]);
		}

		if ((ik->lock[IK::HL] > 0.0) && (ik_weight.y < ik->threshold)) {
			ik->lock[IK::HL] = glm::clamp(ik->lock[IK::HL] - ik->fade, 0.0f, 1.0f);
			ik->lock[IK::TL] = glm::clamp(ik->lock[IK::TL] - ik->fade, 0.0f, 1.0f);
		}

		if ((ik->lock[IK::HR] > 0.0) && (ik_weight.w < ik->threshold)) {
			ik->lock[IK::HR] = glm::clamp(ik->lock[IK::HR] - ik->fade, 0.0f, 1.0f);
			ik->lock[IK::TR] = glm::clamp(ik->lock[IK::TR] - ik->fade, 0.0f, 1.0f);
		}

	}
#endif
}

void PFNNCharacterBase::postUpdateAnim(float deltaTime)
{
	/* Update Past Trajectory */
	for (int i = 0; i < Trajectory::LENGTH / 2; i++)
	{
		trajectory->positions[i] = trajectory->positions[i + 1];
		trajectory->directions[i] = trajectory->directions[i + 1];
		trajectory->rotations[i] = trajectory->rotations[i + 1];
		trajectory->heights[i] = trajectory->heights[i + 1];
		trajectory->gait_stand[i] = trajectory->gait_stand[i + 1];
		trajectory->gait_walk[i] = trajectory->gait_walk[i + 1];
		trajectory->gait_jog[i] = trajectory->gait_jog[i + 1];
		trajectory->gait_crouch[i] = trajectory->gait_crouch[i + 1];
		trajectory->gait_jump[i] = trajectory->gait_jump[i + 1];
		trajectory->gait_bump[i] = trajectory->gait_bump[i + 1];
	}

	/* Update Current Trajectory */
	float stand_amount = powf(1.0f - trajectory->gait_stand[Trajectory::LENGTH / 2], 0.25f);

	glm::vec3 trajectory_update = (trajectory->rotations[Trajectory::LENGTH / 2] * glm::vec3(pfnn->Yp(0), 0, pfnn->Yp(1)));
	trajectory->positions[Trajectory::LENGTH / 2] = trajectory->positions[Trajectory::LENGTH / 2] + stand_amount * trajectory_update;
	trajectory->directions[Trajectory::LENGTH / 2] = glm::mat3(glm::rotate(stand_amount * -pfnn->Yp(2), glm::vec3(0, 1, 0))) * trajectory->directions[Trajectory::LENGTH / 2];
	trajectory->rotations[Trajectory::LENGTH / 2] = glm::mat3(glm::rotate(atan2f(
		trajectory->directions[Trajectory::LENGTH / 2].x,
		trajectory->directions[Trajectory::LENGTH / 2].z), glm::vec3(0, 1, 0)));

#if 0
	/* Collide with walls */
	for (int j = 0; j < areas->num_walls(); j++) {
		glm::vec2 trjpoint = glm::vec2(trajectory->positions[Trajectory::LENGTH / 2].x, trajectory->positions[Trajectory::LENGTH / 2].z);
		glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
		float segdist = glm::length(segpoint - trjpoint);
		if (segdist < areas->wall_width[j] + 100.0) {
			glm::vec2 prjpoint0 = (areas->wall_width[j] + 0.0f) * glm::normalize(trjpoint - segpoint) + segpoint;
			glm::vec2 prjpoint1 = (areas->wall_width[j] + 100.0f) * glm::normalize(trjpoint - segpoint) + segpoint;
			glm::vec2 prjpoint = glm::mix(prjpoint0, prjpoint1, glm::clamp((segdist - areas->wall_width[j]) / 100.0f, 0.0f, 1.0f));
			trajectory->positions[Trajectory::LENGTH / 2].x = prjpoint.x;
			trajectory->positions[Trajectory::LENGTH / 2].z = prjpoint.y;
		}
	}
#endif

	/* Update Future Trajectory */
	for (int i = Trajectory::LENGTH / 2 + 1; i < Trajectory::LENGTH; i++) {
		int w = (Trajectory::LENGTH / 2) / 10;
		float m = fmod(((float)i - (Trajectory::LENGTH / 2)) / 10.0, 1.0);
		trajectory->positions[i].x = (1 - m) * pfnn->Yp(8 + (w * 0) + (i / 10) - w) + m * pfnn->Yp(8 + (w * 0) + (i / 10) - w + 1);
		trajectory->positions[i].z = (1 - m) * pfnn->Yp(8 + (w * 1) + (i / 10) - w) + m * pfnn->Yp(8 + (w * 1) + (i / 10) - w + 1);
		trajectory->directions[i].x = (1 - m) * pfnn->Yp(8 + (w * 2) + (i / 10) - w) + m * pfnn->Yp(8 + (w * 2) + (i / 10) - w + 1);
		trajectory->directions[i].z = (1 - m) * pfnn->Yp(8 + (w * 3) + (i / 10) - w) + m * pfnn->Yp(8 + (w * 3) + (i / 10) - w + 1);
		trajectory->positions[i] = (trajectory->rotations[Trajectory::LENGTH / 2] * trajectory->positions[i]) + trajectory->positions[Trajectory::LENGTH / 2];
		trajectory->directions[i] = glm::normalize((trajectory->rotations[Trajectory::LENGTH / 2] * trajectory->directions[i]));
		trajectory->rotations[i] = glm::mat3(glm::rotate(atan2f(trajectory->directions[i].x, trajectory->directions[i].z), glm::vec3(0, 1, 0)));
	}

	/* Update Phase */
	character->prev_phase = character->phase;
	character->phase = fmod(character->phase + (stand_amount * 0.9f + 0.1f) * 2 * M_PI * pfnn->Yp(3), 2 * M_PI);


	// Update stats
	m_currentSimTime += deltaTime;
	const glm::vec3 rootPosition = trajectory->positions[Trajectory::LENGTH / 2];
	addCurrentRootSample(rootPosition, m_currentSimTime);

	
	addCurrentPoseSample(character->joint_positions, character->joint_mesh_xform, m_currentSimTime);
}
