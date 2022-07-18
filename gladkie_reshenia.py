def my_loss(z, f, num_of_rules, a, h, coeff_list,norm_coeff,cons_coeff,reg_coeff):
    z_a_list = []
    for i in range(num_of_rules):
        z_a_list.append(torch.exp(z[i]) * a[i])

    D_i_vec = torch.zeros(num_of_rules, requires_grad=False)
    for i in range(num_of_rules):
        D_i_vec[i] = torch.sum(z_a_list[i])

    # норма совметсного распредления
    norm_loss = torch.square(1 - torch.sum(D_i_vec))

    consistency_loss = torch.zeros(1, requires_grad=False)

    for i in range(num_of_rules):
        consistency_loss = consistency_loss + \
                           coeff_list[i][0] * torch.sum(
            torch.square((f[i][0] - (torch.sum(z_a_list[i], (1, 2)) / h[i][0]) / D_i_vec[i]))) + \
                           coeff_list[i][1] * torch.sum(
            torch.square((f[i][1] - (torch.sum(z_a_list[i], (0, 2)) / h[i][1]) / D_i_vec[i]))) + \
                           coeff_list[i][2] * torch.sum(
            torch.square((f[i][2] - (torch.sum(z_a_list[i], (0, 1)) / h[i][2]) / D_i_vec[i])))

    reg_loss = torch.zeros(1, requires_grad=False)
    for i in range(num_of_rules):
        reg_loss += torch.norm(torch.exp(z[i]))

    # возвращаем суммарную ошибку чтобы брать градиент, степень согласованности с экспертом и норму
    #
    loss = cons_coeff * consistency_loss + norm_coeff*norm_loss+reg_coeff * reg_loss
    return loss, consistency_loss, norm_loss,reg_loss




def train_and_save_z(f, rules, train_dict,previous_z_lists, plot_gradien_loss, a, h, coeff_list, print_tmp_cons_and_loss):
    train_info = {}
    # инициализируем tmp_z
    num_of_rules = 9
    dimension = 3

    z = []

    for rule in rules:

        size_of_pth_tensor = []
        for distribution in rules[rule]:
            size_of_pth_tensor.append(rules[rule][distribution]["N"])
        z_np = (-5)*torch.rand(size=tuple(size_of_pth_tensor))
        z.append(z_np.clone().detach().requires_grad_(True))
        # z.append(torch.tensor(z_np, requires_grad=True, device=device))
    previous_z_lists_consts= []
    for i in range(len(previous_z_lists)):
        previous_z_lists_consts.append([])
        for j in range(len(previous_z_lists[i])):
            previous_z_lists_consts[i].append(torch.from_numpy(previous_z_lists[i][j]).requires_grad_(False))


    optimizer = torch.optim.Adam(z, train_dict["lr"], [0.5, 0.7])
    loss_vec = np.zeros(train_dict["num_of_epochs"], )
    consyst_vec = np.zeros(train_dict["num_of_epochs"], )
    norm_vec = np.zeros(train_dict["num_of_epochs"], )
    reg_vec = np.zeros(train_dict["num_of_epochs"], )

    norm_coeff = 1
    cons_coeff = 1
    reg_coeff = 1


    last_lr =train_dict["lr"]


    ignoring_norm = False

    last_index_for_plot = 0

    max_index = 10

    for i in range(train_dict["num_of_epochs"]):
        if i in train_dict:
            for g in optimizer.param_groups:
                g['lr'] = train_dict[i]["lr"]
                last_lr = train_dict[i]["lr"]

        if i > max_index:

            if ignoring_norm == False:
                last_mean_loss = np.mean(loss_vec[i - 10:i])
                last_loss = loss_vec[i - 1]
                if np.log10(last_loss) < np.log10(last_mean_loss) * 0.9:
                    last_lr = last_lr * 0.99
                    for g in optimizer.param_groups:
                        g['lr'] = last_lr
                        # print("\n new_lr {}".format(last_lr))
            if ignoring_norm == True:
                last_mean_norm_loss = np.mean(norm_vec[i - 10:i])
                last_norm_loss = norm_vec[i - 1]
                if np.log10(last_norm_loss) < np.log10(last_mean_norm_loss) * 1.1:

                    last_lr = last_lr * 0.9

                    for g in optimizer.param_groups:
                        g['lr'] = last_lr

        optimizer.zero_grad()
        loss, consistency, norm, reg =my_loss(z, f, num_of_rules, a, h, coeff_list,norm_coeff=norm_coeff,cons_coeff=cons_coeff,reg_coeff=reg_coeff)

        norm_loss = float(norm.cpu().detach().numpy())
        cons_loss = float(consistency.cpu().detach().numpy())
        reg_loss = float(reg.cpu().detach().numpy())

        if norm_loss < 1e-3:
            norm_coeff = 0
            ignoring_norm = True
        else:
            norm_coeff = 1 / norm_loss
            ignoring_norm = False

        cons_coeff = 1 / cons_loss
        reg_coeff = 1 / reg_loss

        if print_tmp_cons_and_loss == True:
            print("\r>>   {}% consistency: {}   norm: {} reg {}".format(np.floor((i + 1) / train_dict["num_of_epochs"] * 100),
                                                                 cons_loss,
                                                                 norm_loss,
                                                                 reg_loss),
                                                                end='')
        loss_for_plot = float(loss.cpu().detach().numpy())

        loss_vec[i] = loss_for_plot
        consyst_vec[i] = cons_loss
        norm_vec[i] = norm_loss
        reg_vec[i] = reg_loss

        if(cons_loss < 0.009 and norm_loss <0.001 and reg_loss < 0.007):
            last_index_for_plot= i
            break

        loss.backward()
        optimizer.step()

    for i in range(len(z)):
        z[i] = torch.exp(z[i])

    if plot_gradien_loss == True:
        fig_loss, axs_loss = plt.subplots(1, 4)
        if last_index_for_plot == 0:
            last_index_for_plot = train_dict["num_of_epochs"]-1
        loss_line, = axs_loss[0].plot(loss_vec[:last_index_for_plot])
        axs_loss[0].set_title("loss")
        axs_loss[0].set_yscale("log")
        consistency_line, = axs_loss[1].plot(consyst_vec[:last_index_for_plot])
        axs_loss[1].set_title("consistency")
        axs_loss[1].set_yscale("log")
        norm_line, = axs_loss[2].plot(norm_vec[:last_index_for_plot])
        axs_loss[2].set_title("norm")
        axs_loss[2].set_yscale("log")

        reg_line, = axs_loss[3].plot(reg_vec[:last_index_for_plot])
        axs_loss[3].set_title("reg")
        axs_loss[3].set_yscale("log")

        plt.show(block=True)

    train_info.update({"last_consistency": consyst_vec[-1]})
    train_info.update({"last_norm": norm_vec[-1]})
    return z, train_info