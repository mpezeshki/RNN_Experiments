from datasets import single_bouncing_ball, save_as_gif

X = single_bouncing_ball(10, 10, 200, 15, 2)
save_as_gif(X[0, :, 0, :].reshape(200, 15, 15))
