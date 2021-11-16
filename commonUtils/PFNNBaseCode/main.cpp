#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
//#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "PFNNCharacter.h"
#include "PFNNBaseCode.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);

class PFNNCharacterPythonWrapper : public PFNNCharacterBase 
{
public:
    PFNNCharacterPythonWrapper(const std::string& name) : PFNNCharacterBase(name)
    {
        m_poseBuffer.resize({Character::JOINT_NUM*3});

        m_poseStats_minVelBuffer.resize({Character::JOINT_NUM});
        m_poseStats_maxVelBuffer.resize({Character::JOINT_NUM});
        m_poseStats_avgVelBuffer.resize({Character::JOINT_NUM});
        m_poseStats_maxYawBuffer.resize({Character::JOINT_NUM});
        m_poseStats_avgYawBuffer.resize({Character::JOINT_NUM});

        m_secondHiddenLayerBuffer.resize({ PFNN::HDIM });
    }

    // These are functions used by the Python wrapper that's why they need some kind of conversion of data!!!
	// Get the full pose coordinates
	py::array_t<float>& getCurrentPose() const
	{
		
        float* poseBuffer_ptr = (float*)m_poseBuffer.request().ptr;
		/*if (s_poseBuffer.capacity() < )
		{
			s_poseBuffer.resize(Character::JOINT_NUM*3);
		}
        */

		for (int i = 0; i < Character::JOINT_NUM; i++)
		{
			const glm::vec3& pos = getJointPos(i);
			poseBuffer_ptr[i*3 + 0] = pos.x;
			poseBuffer_ptr[i*3 + 1] = pos.y;
			poseBuffer_ptr[i*3 + 2] = pos.z;
		}

		return m_poseBuffer;        
	}

    ////////// CODE FOR GETTING SAMPLE POSE STATS
    ////////////////////////////////////////////////////////
	py::array_t<float>& getCurrentPose_Stats_minVel() const
	{
	    float* buff = (float*)m_poseStats_minVelBuffer.request().ptr;
	    const SamplePoseStats& stats = getCurrentPoseStats();
	    memcpy(buff, stats.min_vel, sizeof(float) * Character::JOINT_NUM);
	    return m_poseStats_minVelBuffer;
	}

	py::array_t<float>& getCurrentPose_Stats_avgVel() const
	{
	    float* buff = (float*)m_poseStats_avgVelBuffer.request().ptr;
	    const SamplePoseStats& stats = getCurrentPoseStats();
	    memcpy(buff, stats.avg_vel, sizeof(float) * Character::JOINT_NUM);
	    return m_poseStats_avgVelBuffer;
	}

	py::array_t<float>& getCurrentPose_Stats_maxVel() const
	{
	    float* buff = (float*)m_poseStats_maxVelBuffer.request().ptr;
	    const SamplePoseStats& stats = getCurrentPoseStats();
	    memcpy(buff, stats.max_vel, sizeof(float) * Character::JOINT_NUM);
	    return m_poseStats_maxVelBuffer;
	}

	py::array_t<float>& getCurrentPose_Stats_maxYaw() const
	{
	    float* buff = (float*)m_poseStats_maxYawBuffer.request().ptr;
	    const SamplePoseStats& stats = getCurrentPoseStats();
	    memcpy(buff, stats.max_yaw, sizeof(float) * Character::JOINT_NUM);
	    return m_poseStats_maxYawBuffer;
	}

	py::array_t<float>& getCurrentPose_Stats_avgYaw() const
	{
	    float* buff = (float*)m_poseStats_avgYawBuffer.request().ptr;
	    const SamplePoseStats& stats = getCurrentPoseStats();
	    memcpy(buff, stats.avg_yaw, sizeof(float) * Character::JOINT_NUM);
	    return m_poseStats_avgYawBuffer;
	}

	py::array_t<float>& getPrevFrame_SecondHiddenLayerOutput() const
    {
        float* buff = (float*)m_secondHiddenLayerBuffer.request().ptr;
        memcpy(buff, pfnn->H1.data(), sizeof(float) * pfnn->H1.size());
        return m_secondHiddenLayerBuffer;
    }

    float getCurrentPose_Stats_maxOfMaxYaws() const
    {
        const SamplePoseStats& stats = getCurrentPoseStats();
        return stats.maximum_max_yaws;
    }

    float getCurrentPose_Stats_maxOfAvgYaws() const
    {
        const SamplePoseStats& stats = getCurrentPoseStats();
        return stats.maximum_avg_yaws;
    }

    float getCurrentPose_Stats_avgOfMaxYaws() const
    {
        const SamplePoseStats& stats = getCurrentPoseStats();
        return stats.avg_max_yaws;
    }

    float getCurrentPose_Stats_avgOfAvgYaws() const
    {
        const SamplePoseStats& stats = getCurrentPoseStats();
        return stats.avg_avg_yaws;
    }
    ////////////////////////////////////////////////////////


    float getPrevFrame_PhaseUsed() const {
        return character->prev_phase;
    }

    float getCurrentFrame_Phase() const { return character->phase; }

    float getSecondHiddenLayerSize() const { return  PFNN::HDIM; }

	const std::vector<int>& getJointParents() const
	{
		return character->joint_parents;
	}

    void resetPosAndOrientation(const float x, const float y, const float z, const float dirX, const float dirZ)
    {
        PFNNCharacterBase::resetPosAndOrientation(glm::vec2(x, z), glm::vec2(dirX, dirZ));
    }

    void setTargetPosition(const float x, const float z)
    {
        PFNNCharacterBase::setTargetPosition(glm::vec3(x, 0, z));
    }

    py::tuple getCurrent2DPos() const
    {
        const glm::vec3 pos = PFNNCharacterBase::getTrajectoryPos(PFNNCharacterBase::getTrajectoryLen() / 2);
        return py::make_tuple(pos.x, pos.z);
    }

protected:
    // To avoid per frame allocation
    mutable py::array_t<float> m_poseBuffer;//(Character::JOINT_NUM*3);
    mutable py::array_t<float> m_poseStats_minVelBuffer;
    mutable py::array_t<float> m_poseStats_maxVelBuffer;
    mutable py::array_t<float> m_poseStats_avgVelBuffer;
    mutable py::array_t<float> m_poseStats_maxYawBuffer;
    mutable py::array_t<float> m_poseStats_avgYawBuffer;
    mutable py::array_t<float> m_secondHiddenLayerBuffer; //PFNN::HDIM
};


PYBIND11_MODULE(pfnncharacter, m) 
{

    py::bind_vector<std::vector<int>>(m, "VectorInt");
    py::bind_vector<std::vector<float>>(m, "VectorFloat");

    py::class_<PFNNCharacterPythonWrapper>(m, "PFNNCharacter")
        .def(py::init<const std::string&>())
        .def("init", &PFNNCharacterPythonWrapper::init)
        .def("getName", &PFNNCharacterPythonWrapper::getName)
        .def("resetPosAndOrientation", &PFNNCharacterPythonWrapper::resetPosAndOrientation)
        .def("getCurrentPose", &PFNNCharacterPythonWrapper::getCurrentPose, py::return_value_policy::reference)

        .def("getCurrentPose_Stats_minVel", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_minVel, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_maxVel", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_maxVel, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_avgVel", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_avgVel, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_maxYaw", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_maxYaw, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_avgYaw", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_avgYaw, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_maxOfMaxYaws", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_maxOfMaxYaws, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_maxOfAvgYaws", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_maxOfAvgYaws, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_avgOfMaxYaws", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_avgOfMaxYaws, py::return_value_policy::reference)
        .def("getCurrentPose_Stats_avgOfAvgYaws", &PFNNCharacterPythonWrapper::getCurrentPose_Stats_avgOfAvgYaws, py::return_value_policy::reference)

        .def("getJointParents", &PFNNCharacterPythonWrapper::getJointParents, py::return_value_policy::reference)
        .def("setTargetPosition", &PFNNCharacterPythonWrapper::setTargetPosition)
        .def("getCurrent2DPos", &PFNNCharacterPythonWrapper::getCurrent2DPos)
        .def("updateAnim", &PFNNCharacterPythonWrapper::updateAnim)
        .def("postUpdateAnim", &PFNNCharacterPythonWrapper::postUpdateAnim)
        .def("getNumJoints", &PFNNCharacterPythonWrapper::getNumJoints)
        .def("getJointPos", &PFNNCharacterPythonWrapper::getJointPos)
        .def("getJointParent", &PFNNCharacterPythonWrapper::getJointParent)
        .def("setDesiredSpeed", &PFNNCharacterPythonWrapper::setDesiredSpeed)
        .def("getTrajectoryLen", &PFNNCharacterPythonWrapper::getTrajectoryLen)
        .def("getTrajectorySampleFactor", &PFNNCharacterPythonWrapper::getTrajectorySampleFactor)
        .def("getTrajectoryPos", &PFNNCharacterPythonWrapper::getTrajectoryPos)
        .def("getCurrentAvgSpeed", &PFNNCharacterPythonWrapper::getCurrentAvgSpeed)
        .def("getTargetReachedThreshold", &PFNNCharacterPythonWrapper::getTargetReachedThreshold)
        //.def("setTargetReachedThreshold", &PFNNCharacterPythonWrapper::setTargetReachedThreshold)
        .def("getCurrentFrame_Phase", &PFNNCharacterPythonWrapper::getCurrentFrame_Phase)
        .def("getPrevFrame_Phase", &PFNNCharacterPythonWrapper::getPrevFrame_PhaseUsed)
        .def("getPrevFrame_SecondHiddenLayerOutput", &PFNNCharacterPythonWrapper::getPrevFrame_SecondHiddenLayerOutput, py::return_value_policy::reference)
        .def("getSecondHiddenSizeLayerOutput", &PFNNCharacterPythonWrapper::getSecondHiddenLayerSize)
        .def("getInternalSystemSpeed", &PFNNCharacterPythonWrapper::getInternalSystemSpeed);

    

    m.doc() = R"pbdoc(
        Pybind11 for pfnncharacter
        -----------------------

        .. currentmodule:: pfnncharacter
.def("init", &PFNNCharacterBase::init)
        .. autosummary::
           :toctree: _generate

           pfnncharacter character and its functions
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

#ifdef BUILD_AS_EXECUTABLE
int main()
{
    PFNNCharacterBase  ch("ciprian");
    ch.init("");
    ch.resetPosAndOrientation(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));

    printf("test works");
    return 0;
}
#endif