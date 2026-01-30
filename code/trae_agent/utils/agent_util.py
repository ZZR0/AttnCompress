import os
import json

TIME_OUT_LABEL= ' seconds. Partial output:'

def remove_patches_to_tests(model_patch):
    """
    Remove any changes to the tests directory from the provided patch.
    This is to ensure that the model_patch does not disturb the repo's
    tests when doing acceptance testing with the `test_patch`.
    """
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    is_tests = False

    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            to = pieces[-1]
            if to.startswith("b/") and any(
                x in to for x in [
                    "/test/", "/tests/", "/testing/", "/test_",
                    ".tests.", ".test.", "_test_", "_tests_", "_test.", "_tests.", ".spec.ts",
                    "/tox.ini", "/Cargo.lock", "/package.json", "/package-lock.json", "/pom.xml",
                ]
            ):
                is_tests = True
            else:
                is_tests = False

        if not is_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)

def save_patches(instance_id, patches_path, patches):
    trial_index = 1

    def get_unique_filename(patches_path, trial_index):
        filename = f"{instance_id}_{trial_index}.patch"
        while os.path.exists(os.path.join(patches_path, filename)):
            trial_index += 1
            filename = f"{instance_id}_{trial_index}.patch"
        return filename

    patch_file = get_unique_filename(patches_path, trial_index)

    clean_patch = patches #remove_patches_to_tests(patches)

    with open(os.path.join(patches_path, patch_file), 'w') as file:
        file.write(clean_patch)

    print(f"Patches saved in {patches_path}/{patch_file}")
    print(clean_patch)
    return f"{patches_path}/{patch_file}"

def save_trajectory(instance_id, traj_dir, trajectory):
    trial_index = 1

    def get_unique_filename(traj_dir, trial_index):
        filename = f"{instance_id}_{trial_index}.json"
        while os.path.exists(os.path.join(traj_dir, filename)):
            trial_index += 1
            filename = f"{instance_id}_{trial_index}.json"
        return filename
    
    traj_file = get_unique_filename(traj_dir, trial_index)
    trajectory_json = json.dumps(trajectory, indent=4, sort_keys=False, ensure_ascii=False)
    with open(os.path.join(traj_dir, traj_file), 'a', encoding='utf-8') as file:
        file.write(f"{trajectory_json}\n")
