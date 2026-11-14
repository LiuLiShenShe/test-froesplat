"""Representative coverage must not amplify rare camera contribution mass."""
from types import SimpleNamespace
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.historical.camera_quadrature import representative_weights


def scene():
    views=[]
    for x in [0]*7+[1]:
        E=np.eye(4);E[0,3]=-x;views.append(dict(E=E))
    return SimpleNamespace(views=views)


def test_dense_camera_population_and_rare_view_have_original_mass():
    weights=representative_weights(scene(),[0,7])
    np.testing.assert_allclose(weights,[7/8,1/8])
    # Rare viewpoint sees an artifact overlapping the leaf; uniform FPS
    # averaging overstates that foreground attribution by more than2×.
    fg=np.array([1.,10.])
    assert np.isclose(weights@fg,(7*1+10)/8)
    assert fg.mean()>2*(weights@fg)


def test_full_camera_evaluation_is_exact_even_with_repeated_pose_features():
    np.testing.assert_allclose(representative_weights(scene(),list(range(8))),np.ones(8)/8)


def test_weights_follow_requested_camera_order():
    np.testing.assert_allclose(representative_weights(scene(),[7,0]),[1/8,7/8])
