"""Create latents for 3DIdent dataset."""
import os
import numpy as np
import sys 
sys.path.append("/cluster/work/vogtlab/Group/palumboe/cl-ica_mm")
import spaces
import latent_spaces
import argparse
import spaces_utils
import shutil

def return_uniform_probs(n_opt):
    return [1./n_opt for i in range(n_opt)]

def text_rendura(latents, prompt):
    return prompt.format(**{"COL": latents[0], "POSY": latents[1], "POSX": latents[2], "OBJ": 'teapot'})

map_colorclasses = {0: 'red',
                    1: 'yellow',
                    2: 'green',
                    3: 'cyan',
                    4: 'blue',
                    5: 'magenta'}

map_positions_y = {2: 'right',
                   1: 'center',
                   0: 'left'}

map_positions_x = {2: 'bottom',
                   1: 'mid',
                   0: 'top'}

#map_classes = {0: 'teapot',
#               1: 'hare',
#               2: 'dragon',
#               3: 'cow',
#               4: 'armadillo',
#               5: 'horse',
#               6: 'head'}

n_color_classes = len(map_colorclasses.keys())
n_positions_x = len(map_positions_x.keys())
n_positions_y = len(map_positions_y.keys())
#n_classes = len(map_classes.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", default=1000000, type=int)
    parser.add_argument("--n-objects", default=1, type=int)
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--n-batches", default=2, type=int)
    parser.add_argument("--batch_index", default=1, type=int)
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--rotation-and-color-only", action="store_true")
    parser.add_argument("--rotation-only", action="store_true")
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--fixed-spotlight", action="store_true")
    parser.add_argument("--non-periodic-rotation-and-color", action="store_true")

    args = parser.parse_args()

    print(args)

    assert not (
        args.position_only and args.rotation_and_color_only
    ), "Only either position-only or rotation-and-color-only can be set"

    os.makedirs(args.output_folder, exist_ok=True)

    """
    render internally assumes the variables form these value ranges:
    
    per object:
        0. x position in [-3, -3]
        1. y position in [-3, -3]
        2. z position in [-3, -3]
        3. alpha rotation in [0, 2pi]
        4. beta rotation in [0, 2pi]
        5. gamma rotation in [0, 2pi]
        6. theta spot light in [0, 2pi]
        7. hue object in [0, 2pi]
        8. hue spot light in [0, 2pi]
    
    per scene:
        9. hue background in [0, 2pi]
    """

    n_angular_variables = args.n_objects * 6 + 1
    n_non_angular_variables = args.n_objects * 3

    if args.non_periodic_rotation_and_color:
        s = latent_spaces.LatentSpace(
            spaces.NBoxSpace(n_non_angular_variables + n_angular_variables),
            lambda space, size, device: space.uniform(size, device=device),
            None,
        )

    else:
        s = latent_spaces.ProductLatentSpace(
            [
                latent_spaces.LatentSpace(
                    spaces.NBoxSpace(n_non_angular_variables),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
                latent_spaces.LatentSpace(
                    spaces.NSphereSpace(n_angular_variables + 1),
                    lambda space, size, device: space.uniform(size, device=device),
                    None,
                ),
            ]
        )

    
    # Hand-engineered propmpts

    text_prompts =  {0: "An {OBJ} of {COL} color is visible, positioned in the {POSY}-{POSX} of the image." ,
                    1: "A {COL} {OBJ} is on the {POSY}-{POSX} of the image.",
                    2: "The {POSY}-{POSX} of the image represents a {COL} colored {OBJ}.",
                    3: "On the {POSY}-{POSX} of the picture there is a {OBJ} in {COL} color.",
                    4: "On the {POSY}-{POSX} of the image, there is a {COL} object, in the form of a {OBJ}." }
    n_text_prompts = len(text_prompts.keys())

    s_t = latent_spaces.LatentSpace(
                spaces.DiscreteSpace(1, probs_=return_uniform_probs(n_text_prompts)),
                lambda space, size, device: space.sample_from_distribution(size, device=device),
                None,
            )

    prompts = s_t.sample_marginal(args.n_points, device="cpu").numpy()
    prompts_actval = np.vectorize(text_prompts.get)(prompts)

    raw_latents = s.sample_marginal(args.n_points, device="cpu").numpy()

    if True: # TODO Flag Position is shared
        s_position = latent_spaces.ProductLatentSpace(
            [
                latent_spaces.LatentSpace(
                    spaces.DiscreteSpace(1, probs_=return_uniform_probs(3)),
                    lambda space, size, device: space.sample_from_distribution(size, device=device),
                    None,
                ),  # Position x
                latent_spaces.LatentSpace(
                    spaces.DiscreteSpace(1, probs_=return_uniform_probs(3)),
                    lambda space, size, device: space.sample_from_distribution(size, device=device),
                    None,
                ),  # Position y
                #latent_spaces.LatentSpace(
                #    spaces.DiscreteSpace(1, probs_=return_uniform_probs(3)),
                #    lambda space, size, device: space.sample_from_distribution(size, device=device),
                #    None,
                #),  # Position z
            ]
        )
        raw_latents_position = s_position.sample_marginal(args.n_points, device="cpu").numpy()
        raw_latents_position[0,0] = 0
        raw_latents_position[1,0] = 1
        raw_latents_position[2,0] = 2
        raw_latents_position[3,0] = 0
        raw_latents_position[4,0] = 1
        raw_latents_position[5,0] = 2
        raw_latents_position[6,0] = 0
        raw_latents_position[7,0] = 1
        raw_latents_position[8,0] = 2
        raw_latents_position[0,1] = 0
        raw_latents_position[1,1] = 0
        raw_latents_position[2,1] = 0
        raw_latents_position[3,1] = 1
        raw_latents_position[4,1] = 1
        raw_latents_position[5,1] = 1
        raw_latents_position[6,1] = 2
        raw_latents_position[7,1] = 2
        raw_latents_position[8,1] = 2
        positions_x_actval = np.vectorize(map_positions_x.get)(raw_latents_position[:, 0])
        positions_y_actval = np.vectorize(map_positions_y.get)(raw_latents_position[:, 1])
        position_latents_new = raw_latents_position[:, :]
        position_latents_new = (position_latents_new - 1) * 1.5

    if True: # TODO Flag Hue Color is shared
        s_hueobject = latent_spaces.LatentSpace(
            spaces.DiscreteSpace(1, probs_=return_uniform_probs(6)),
            lambda space, size, device: space.sample_from_distribution(size, device=device),
            None,
        )
        hue_object_latents_new = s_hueobject.sample_marginal(args.n_points, device="cpu").numpy()
        color_variables_actval = np.vectorize(map_colorclasses.get)(hue_object_latents_new[:,0])
        hue_object_latents_new = (hue_object_latents_new * 2 + 1) * np.pi / 6


    if args.position_only or args.rotation_and_color_only:
        assert args.n_objects == 1, "Only one object is supported for fixed variables"

    if args.non_periodic_rotation_and_color:
        if args.position_only:
            raw_latents[:, n_non_angular_variables:] = np.array(
                [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
            )
        if args.rotation_and_color_only or args.rotation_only or args.color_only:
            raw_latents[:, :n_non_angular_variables] = np.array([0, 0, 0])
        if args.rotation_only:
            # additionally fix color
            raw_latents[:, -3:] = np.array([-1, 0, 1.0])
        if args.color_only:
            # additionally fix rotation
            raw_latents[
                :, n_non_angular_variables : n_non_angular_variables + 4
            ] = np.array([-1, -0.5, 0.5, 1.0])

        if args.fixed_spotlight:
            # assert not args.rotation_only
            raw_latents[:, [-2, -4]] = np.array([0.0, 0.0])

        # the raw latents will later be used for the sampling process
        np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # get rotation and color latents from large vector
        rotation_and_color_latents = raw_latents[:, n_non_angular_variables:]
        rotation_and_color_latents *= np.pi / 2

        position_latents = raw_latents[:, :n_non_angular_variables]
        position_latents *= 3
        position_latents_old_z = position_latents[:,-2:-1]


    else:
        if args.position_only:
            spherical_fixed_angular_variables = np.array(
                [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, 0, 1.5 * np.pi]
            )
            cartesian_fixed_angular_variables = spaces_utils.spherical_to_cartesian(
                1, spherical_fixed_angular_variables
            )
            raw_latents[:, n_non_angular_variables:] = cartesian_fixed_angular_variables
        if args.rotation_and_color_only:
            fixed_non_angular_variables = np.array([0, 0, 0])
            raw_latents[:, :n_non_angular_variables] = fixed_non_angular_variables

        np.save(os.path.join(args.output_folder, "raw_latents.npy"), raw_latents)

        # convert angular latents from cartesian to angular representation
        rotation_and_color_latents = spaces_utils.cartesian_to_spherical(
            raw_latents[:, n_non_angular_variables:]
        )[1]
        # map all but the last latent from [0,pi] to [0, 2pi]
        rotation_and_color_latents[:, :-1] *= 2

        position_latents = raw_latents[:, :n_non_angular_variables]
        # map z coordinate from -1,+1 to 0,+1
        position_latents[:, 2:n_non_angular_variables:3] = (
            position_latents[:, 2:n_non_angular_variables:3] + 1
        ) / 2.0
        position_latents *= 0
        position_latents_old_z = position_latents[:,-2:-1]

    rotation_and_color_latents[:,:] =rotation_and_color_latents[:,-3:-2]*0
    rotation_and_color_latents[:,-3:-2] = hue_object_latents_new
    latents = np.concatenate((position_latents_new*2, position_latents_old_z, rotation_and_color_latents), 1)

    n_samples = len(latents)
    #indices = np.array_split(np.arange(n_samples), args.n_batches)[args.batch_index]
    output_tex_folder = os.path.join(args.output_folder, "text")
    if os.path.exists(output_tex_folder):
        shutil.rmtree(output_tex_folder)
        os.makedirs(
            output_tex_folder
        )
    else:
        os.makedirs(
            output_tex_folder
        )
    for idx in range(n_samples):
        print(latents[idx])
        print(color_variables_actval[idx], positions_x_actval[idx], positions_y_actval[idx])
        output_filename = os.path.join(
            output_tex_folder,
            f"{str(idx).zfill(int(np.ceil(np.log10(n_samples))))}.txt",
        )
        if os.path.exists(output_filename):
            print("Skipped file", output_filename)
            continue
        else:
            with open(output_filename, 'w+') as f:
                prompt = prompts_actval[idx][0]
                tuple_for_prompt = (color_variables_actval[idx], positions_x_actval[idx], positions_y_actval[idx] )
                f.write(text_rendura(tuple_for_prompt, prompt))


    reordered_transposed_latents = []
    for n in range(args.n_objects):
        reordered_transposed_latents.append(latents.T[n * 3 : n * 3 + 3])
        reordered_transposed_latents.append(
            latents.T[
                n_non_angular_variables + n * 6 : n_non_angular_variables + n * 6 + 6
            ]
        )

    reordered_transposed_latents.append(latents.T[-1].reshape(1, -1))
    reordered_latents = np.concatenate(reordered_transposed_latents, 0).T

    # the latents will be used by the rendering process to generate the images
    np.save(os.path.join(args.output_folder, "latents.npy"), reordered_latents)


if __name__ == "__main__":
    main()
