import matplotlib.pyplot as plt


def plot_line_bars(data: list, legends: list = [], x_ticks: list = None, save_path=""):
    a = data[0]
    for i in range(1, len(data)):
        if len(data[i]) != len(a):
            assert False
    assert len(legends) == 0 or (len(legends) != 0 and len(legends) == len(data))
    fig, ax = plt.subplots()
    print(data)
    for i in range(len(data)):
        if len(legends) != 0:
            ax.plot(data[i])
    ax.legend(legends)
    if save_path != "":
        fig.savefig(fname=save_path, bbox_inches="tight", pad_inches=0)


def get_global_and_local_change(client_id, ori_data, col_name):

    loss_all = ori_data.query('client_id == "{}"'.format(client_id))
    flatten_id_list = list(loss_all["flatten_id"])
    zero_gap = 0
    global_epoch = 0
    loss = list(
        ori_data.query(
            'client_id == "{}" and  flatten_id==0 and global_epoch=={}'.format(
                client_id, 0
            )
        )[col_name]
    )[0]
    global_change = [loss]
    for i in range(1, len(flatten_id_list)):
        if flatten_id_list[i] == 0:
            global_epoch += 1
            last_loss = loss
            loss = list(
                ori_data.query(
                    'client_id == "{}" and  flatten_id==0 and global_epoch=={}'.format(
                        client_id, global_epoch
                    )
                )[col_name]
            )[0]
            for j in range(zero_gap):
                global_change.append(
                    last_loss + (j + 1) / (zero_gap + 1) * (loss - last_loss)
                )
            zero_gap = 0
            global_change.append(loss)
        else:
            zero_gap += 1
    local_change = list(loss_all[col_name])
    return global_change, local_change
