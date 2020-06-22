import numpy as np
# Helper functions
import helper
from attack import PixelAttacker
# Custom Networks
from differential_evolution import differential_evolution


def single_pixel_attack(images,labels,class_names,img_id, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False, plot=False, dimensions=(32, 32)):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else labels[img_id, 0]

    ack = PixelAttacker([model],(images,labels),class_names)

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    dim_x, dim_y = dimensions
    bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return ack.predict_classes(xs, images[img_id], target_class, model, target is None)

    def callback_fn(x, convergence):
        return ack.attack_success(x, images[img_id], target_class, model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = helper.perturb_image(attack_result.x, images[img_id])[0]
    prior_probs = model.predict(np.array([images[img_id]]))[0]
    predicted_probs = model.predict(np.array([attack_image]))[0]
    predicted_class = np.argmax(predicted_probs)
    actual_class = labels[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    if plot:
        helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [attack_image, model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]
