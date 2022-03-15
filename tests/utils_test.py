import sp_nomcpr
import fpsim as fp
import fp_analyses as fa


# Toggling usage of a particular contraceptive off
# Pars from sp_nomcpr.make_pars
def toggle_to_zero(pars, method, age_classification=None):
    method_index = self.pars['methods']['map'][method]
    prob_matrix = self.pars['methods']["probs_matrix"]

    def toggle_index_off(method_array, method_index):
        added_value = method_array[method_index]
        new_list = deepcopy(method_array)
        new_list[0] += added_value
        new_list[method_index] = 0
        return new_list
    
    new_probs_matrix = {}
    # Switch the index element of each array in the dictionary (format  is age_key: array)
    for age_key, value in prob_matrix.items():
        if age_classification is None:
            new_2d_array = [0] * 10
            prob_arrays = self.pars['methods']["probs_matrix"][age_key]
            for index, prob_array in enumerate(prob_arrays): # adding prob of transitioning to method to transitioning to none, setting prob of transition to method to 0 for all methods
                if index != method_index:
                    new_2d_array[index] = toggle_index_off(prob_array, method_index)
                else:
                    new_method_list = [0] * 10
                    new_method_list [0] = 1.0
                    new_2d_array[index] = new_method_list
        new_probs_matrix[age_key] = pl.array(new_2d_array)
    self.pars['methods']["probs_matrix"] = new_probs_matrix