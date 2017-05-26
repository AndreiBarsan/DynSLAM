

#ifndef INFINITAM_SPARSESCENEFLOWCOMPONENT_H
#define INFINITAM_SPARSESCENEFLOWCOMPONENT_H


#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"

namespace InstRecLib {

	/// \brief Interface for components which can compute sparse scene flow from a scene view.
	class SparseSFProvider {

	private:

	public:

		// TODO(andrei): Data holder for SF computation result.
		// TODO(andrei): Pass in time-aware data holder, since we need at least two time frames in order
		// to compute scene flow.
		virtual void ComputeSparseSceneFlow(const ITMLib::Objects::ITMView *view) = 0;

	};
}

#endif //INFINITAM_SPARSESCENEFLOWCOMPONENT_H
