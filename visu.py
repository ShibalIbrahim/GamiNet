import csaps
from scipy.interpolate import interp1d

visu_info = {}
input_grid_num = 100
active_univariate_index, active_interaction_index, beta, gamma, componment_scales = model.get_active_subnets()
        
for indice in range(model.input_num):

    if indice < model.numerical_input_num:

        subnet = model.subnet_blocks.subnets[indice]
        feature_name = list(model.variables_names)[model.noncateg_index_list[indice]]

        sx = model.meta_info[feature_name]["scaler"]
        data_min, data_max = sx.feature_range[0], sx.feature_range[1]
        input_grid_step = (data_max - data_min) / input_grid_num
        subnets_inputs = np.arange(data_min + input_grid_step / 2, data_max, input_grid_step).reshape([-1, 1])
        subnets_outputs = np.sign(beta[indice]) * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy().ravel() * beta[indice]
        subnets_inputs_original = sx.inverse_transform(subnets_inputs).ravel()
        visu_info.update({feature_name: {"type":"continuous",
                                         "inputs":subnets_inputs_original,
                                         "outputs":subnets_outputs}})
    else:

        feature_name = model.categ_variable_list[indice - model.numerical_input_num]
        dummy_gamma = model.categ_blocks.categnets[indice - model.numerical_input_num].categ_bias.numpy()
        norm = model.categ_blocks.categnets[indice - model.numerical_input_num].moving_norm.numpy()
        
        subnets_inputs = np.arange(len(model.meta_info[feature_name]['values']))
        subnets_outputs = np.sign(beta[indice]) * dummy_gamma[:, 0] / norm * beta[indice - model.numerical_input_num]
        subnets_inputs_ticks = model.meta_info[model.categ_variable_list[indice - model.numerical_input_num]]['values']
        visu_info.update({feature_name:{"type":"categorical",
                                        "inputs":subnets_inputs,
                                        "outputs":subnets_outputs,
                                        "xticks":model.meta_info[model.categ_variable_list[indice 
                                                                 - model.numerical_input_num]]['values']}})

for indice in active_interaction_index:    

    inter_net = model.interact_blocks.sub_interacts[indice]
    interact_idxs = model.interaction_list[indice]
    feature_name1 = model.variables_names[interact_idxs[0]]
    feature_name2 = model.variables_names[interact_idxs[1]]
    sx1 = model.meta_info[feature_name1]["scaler"]
    sx2 = model.meta_info[feature_name2]["scaler"]

    data_min, data_max = sx1.feature_range[0], sx1.feature_range[1]
    input_grid_step = (data_max - data_min) / input_grid_num
    numerical_tick_value1 = np.arange(data_min + input_grid_step / 2, data_max, input_grid_step).reshape([-1, 1])
    data_min, data_max = sx2.feature_range[0], sx2.feature_range[1]
    input_grid_step = (data_max - data_min) / input_grid_num
    numerical_tick_value2 = np.arange(data_min + input_grid_step / 2, data_max, input_grid_step).reshape([-1, 1])

    x1, x2 = np.meshgrid(numerical_tick_value1, numerical_tick_value2[::-1])
    response = gamma[indice] * inter_net.apply(tf.cast(np.vstack([x1.ravel(), 
                        x2.ravel()]).T, tf.float32)).numpy().reshape([input_grid_num, input_grid_num]) 

    x1 = train_x[:,interact_idxs[0]]
    x2 = train_x[:,interact_idxs[1]]
    ls1 = []
    for i in range(input_grid_num):
        if i == 0:
            inputs = np.hstack([x1[(x1 >= -1 + i * 2 / input_grid_num) & (x1 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1]), 
                                x2[(x1 >= -1 + i * 2 / input_grid_num) & (x1 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1])])
        else:
            inputs = np.hstack([x1[(x1 > -1 + i * 2 / input_grid_num) & (x1 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1]), 
                                x2[(x1 > -1 + i * 2 / input_grid_num) & (x1 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1])])
        idx, cnts = np.unique(((inputs[:, 1] + 1 / input_grid_num) / (2 / input_grid_num)).astype(int), return_counts=True)
        outputs = np.sum(response[idx, i] * cnts) / np.sum(cnts)
        ls1.append(outputs)
        
    xp = pd.Series(ls1).dropna().index.values
    fp = pd.Series(ls1).dropna().values.ravel()
    x = np.arange(0,input_grid_num)
    sp = csaps.UnivariateCubicSmoothingSpline(xp, fp, smooth=0.05)
    ls1 = sp(x)

    if feature_name1 in visu_info:
        visu_info[feature_name1]["outputs"] = visu_info[feature_name1]["outputs"].ravel() + ls1
                
    response1 = response - ls1
    
    ls2 = []
    for i in range(input_grid_num):
        if i == 0:
            inputs = np.hstack([x1[(x2 >= -1 + i * 2 / input_grid_num) & (x2 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1]), 
                                x2[(x2 >= -1 + i * 2 / input_grid_num) & (x2 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1])])
        else:
            inputs = np.hstack([x1[(x2 > -1 + i * 2 / input_grid_num) & (x2 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1]), 
                                x2[(x2 > -1 + i * 2 / input_grid_num) & (x2 <= -(1 - 2 / input_grid_num) + i * 2 / input_grid_num)].reshape([-1, 1])])
        idx, cnts = np.unique(((inputs[:, 0] + 1 / input_grid_num) / (2 / input_grid_num)).astype(int), return_counts=True)
        outputs = np.sum(response1[i, idx] * cnts) / np.sum(cnts)
        ls2.append(outputs)
        
    xp = pd.Series(ls2).dropna().index.values
    fp = pd.Series(ls2).dropna().values.ravel()
    x = np.arange(0,input_grid_num)
    sp = csaps.UnivariateCubicSmoothingSpline(xp, fp, smooth=0.05)
    ls2 = sp(x)
    if feature_name2 in visu_info:
        visu_info[feature_name2]["outputs"] = visu_info[feature_name2]["outputs"] + ls2

    response2 = response1 - ls2.reshape([-1, 1])
    
    tick_step = int(input_grid_num * 0.2)
    numerical_tick_loc = np.arange(1, input_grid_num, tick_step).reshape([-1, 1])
    x1_tick_loc = numerical_tick_loc.ravel() 
    x2_tick_loc = numerical_tick_loc.ravel()
    x1_real_values = sx1.inverse_transform(numerical_tick_value1).ravel()[x1_tick_loc]
    x2_real_values = sx2.inverse_transform(numerical_tick_value2).ravel()[x2_tick_loc][::-1]

    if (np.max(x1_real_values) - np.min(x1_real_values)) < 0.01:
        x_tick_values1 = np.array([np.format_float_scientific(x1_real_values[i], 
                     precision=1) for i in range(x1_real_values.shape[0])])
    elif (np.max(x1_real_values) - np.min(x1_real_values)) < 10:
        x1_real_values = np.round(x1_real_values, 2)
    elif (np.max(x1_real_values) - np.min(x1_real_values)) < 1000:
        x1_real_values = np.round(x1_real_values).astype(int)
    else:
        x1_real_values = np.array([np.format_float_scientific(x1_real_values[i],
                     precision=1) for i in range(x1_real_values.shape[0])])

    if (np.max(x2_real_values) - np.min(x2_real_values)) < 0.01:
        x2_real_value = np.array([np.format_float_scientific(x2_real_values[i], 
                     precision=1) for i in range(x2_real_values.shape[0])])
    elif (np.max(x2_real_values) - np.min(x2_real_values)) < 10:
        x2_real_values = np.round(x2_real_values, 2)
    elif (np.max(x2_real_values) - np.min(x2_real_values)) < 1000:
        x2_real_values = np.round(x2_real_values).astype(int)
    else:
        x2_real_values = np.array([np.format_float_scientific(x2_real_values[i],
                     precision=1) for i in range(x2_real_values.shape[0])])

    visu_info.update({feature_name1 + " x " + feature_name2: {"type":"pairwise",
                                                              "x1_tick_loc":x1_tick_loc,
                                                              "x2_tick_loc":x2_tick_loc,
                                                              "x1_real_values":x1_real_values,
                                                              "x2_real_values":x2_real_values,
                                                              "outputs": response2}})

cols_per_row=3 
main_density=3
ratio_list = []
for idx, (key, item) in enumerate(visu_info.items()):
    ratio_list.append(np.std(item['outputs']))

ratio_list = np.round(np.array(ratio_list) / np.sum(ratio_list), 4)
sorted_index = np.argsort(ratio_list)
active_index = sorted_index[ratio_list[sorted_index].cumsum()>0.05][::-1]
max_ids = len(active_index)

idx = 0
f = plt.figure(figsize=(6 * cols_per_row, round(max_ids * 6 / cols_per_row)))
for indice in active_index:
    
    key = list(visu_info.keys())[indice]
    item = visu_info[key]
    if item['type'] == "continuous":
        ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                                rowspan=main_density - 1)
        ax1.plot(item['inputs'],item['outputs'])
        ax1.set_ylabel("Score", fontsize=12)

        ax2 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row) + main_density - 1, idx%cols_per_row))

        ax2.hist(sx.inverse_transform(model.tr_x[:,[indice]]), density=True, bins=30)
        ax2.set_ylabel("Density", fontsize=12)
        ax1.set_title(key + " (" + str(np.round(100 * ratio_list[indice], 1)) + "%)")
        idx = idx + 1

for indice in active_index:
    key = list(visu_info.keys())[indice]
    item = visu_info[key]
    if item['type'] == "pairwise":
        ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                               (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                                rowspan=main_density)

        cf = ax1.imshow(item['outputs'])
        plt.xticks(item["x1_tick_loc"], item["x1_real_values"], fontsize=10)
        plt.yticks(item["x2_tick_loc"], item["x2_real_values"], fontsize=10)

        f.colorbar(cf, ax=ax1, format='%0.1f', orientation='horizontal')
        ax1.set_title(key + " (" + str(np.round(100 * ratio_list[indice], 1)) + "%)")
        idx = idx + 1
f.tight_layout()