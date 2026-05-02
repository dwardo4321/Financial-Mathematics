cols = ["Call Price", "Put Price", "Asset Prices"]

        if plot:
            fig = utils.delta_engine_plotter(plot, (11, 8.5), cols, t_0n, out, nrow = 3, ncol = 1)
            utils.data_display(1000)
            return out, fig
        else: