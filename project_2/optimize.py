import optuna
import pathlib
import numpy as np
import torch

from solution import SWAGInference, SWAGScheduler, setup_seeds, cost_function

def objective(trial):
    #threshold = trial.suggest_uniform('threshold', 0.0, 1.0)
    swag_epochs = trial.suggest_int('swag_epochs', 10, 500)
    swag_lr = trial.suggest_loguniform('swag_lr', 1e-5, 1e-1)
    swag_update_freq = trial.suggest_int('swag_update_freq', 1, 10)
    deviation_max_rank = trial.suggest_int('deviation_max_rank', 1, swag_epochs//swag_update_freq)
    bma_samples = trial.suggest_int('bma_samples', 1, 100)

    data_dir = pathlib.Path.cwd()
    model_dir = pathlib.Path.cwd()
    output_dir = pathlib.Path.cwd()

    # Load training data
    train_xs = torch.from_numpy(np.load(data_dir / "train_xs.npz")["train_xs"])
    raw_train_meta = np.load(data_dir / "train_ys.npz")
    train_ys = torch.from_numpy(raw_train_meta["train_ys"])
    train_is_snow = torch.from_numpy(raw_train_meta["train_is_snow"])
    train_is_cloud = torch.from_numpy(raw_train_meta["train_is_cloud"])
    dataset_train = torch.utils.data.TensorDataset(train_xs, train_is_snow, train_is_cloud, train_ys)

    # Load validation data
    val_xs = torch.from_numpy(np.load(data_dir / "val_xs.npz")["val_xs"])
    raw_val_meta = np.load(data_dir / "val_ys.npz")
    val_ys = torch.from_numpy(raw_val_meta["val_ys"])
    val_is_snow = torch.from_numpy(raw_val_meta["val_is_snow"])
    val_is_cloud = torch.from_numpy(raw_val_meta["val_is_cloud"])
    dataset_val = torch.utils.data.TensorDataset(val_xs, val_is_snow, val_is_cloud, val_ys)

    setup_seeds()

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    # Create the model
    model = SWAGInference(
        train_xs=dataset_train.tensors[0],
        model_dir=model_dir,
        swag_epochs=swag_epochs,
        swag_learning_rate=swag_lr,
        swag_update_freq=swag_update_freq,
        deviation_matrix_max_rank=deviation_max_rank,
        bma_samples=bma_samples,
    )

    optimizer = torch.optim.SGD(
        model.network.parameters(),
        lr=model.swag_learning_rate,
        momentum=0.9,
        nesterov=False,
        weight_decay=1e-4,
    )
    
    loss = torch.nn.CrossEntropyLoss(
        reduction="mean",
    )

    lr_scheduler = SWAGScheduler(
        optimizer,
        epochs=model.swag_epochs,
        steps_per_epoch=len(train_loader),
    )

    model.reset_swag_statistics()
    model.update_swag()

    best_cost = np.inf
    model.network.train()
    for epoch in range(model.swag_epochs):
        model.network.train()
        average_loss = 0.0
        average_accuracy = 0.0
        num_samples_processed = 0
        for batch_xs, batch_is_snow, batch_is_cloud, batch_ys in train_loader:
            batch_xs = batch_xs.to(model.device)
            batch_ys = batch_ys.to(model.device)
            optimizer.zero_grad()
            pred_ys = model.network(batch_xs)
            batch_loss = loss(input=pred_ys, target=batch_ys)
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Calculate cumulative average training loss and accuracy
            average_loss = (batch_xs.size(0) * batch_loss.item() + num_samples_processed * average_loss) / (
                num_samples_processed + batch_xs.size(0)
            )
            average_accuracy = (
                torch.sum(pred_ys.argmax(dim=-1) == batch_ys).item()
                + num_samples_processed * average_accuracy
            ) / (num_samples_processed + batch_xs.size(0))
            num_samples_processed += batch_xs.size(0)

        if epoch % model.swag_update_freq == 0:
            model.update_swag()

        # Calculate validation loss and accuracy
        if epoch % 50 == 0 and epoch//model.swag_update_freq > model.deviation_matrix_max_rank:
            model.network.eval()
            with torch.no_grad():
                xs, is_snow, is_cloud, ys = dataset_val.tensors

                pred_prob_all = model.predict_probabilities(xs)
                pred_prob_max, pred_ys_argmax = torch.max(pred_prob_all, dim=-1)
                pred_ys = model.predict_labels(pred_prob_all)

                thresholds = [0.0] + list(torch.unique(pred_prob_max, sorted=True))
                costs = []
                for threshold in thresholds:
                    thresholded_ys = torch.where(pred_prob_max <= threshold, -1 * torch.ones_like(pred_ys), pred_ys)
                    costs.append(cost_function(thresholded_ys, ys).item())
                best_idx = np.argmin(costs)
                if costs[best_idx] < best_cost:
                    best_cost = costs[best_idx]
                    print(f"New best cost: {best_cost} with threshold {thresholds[best_idx]}")

                trial.report(costs[best_idx], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
    model.network.eval()
    with torch.no_grad():
        xs, is_snow, is_cloud, ys = dataset_val.tensors

        pred_prob_all = model.predict_probabilities(xs)
        pred_prob_max, pred_ys_argmax = torch.max(pred_prob_all, dim=-1)
        pred_ys = model.predict_labels(pred_prob_all)

        thresholds = [0.0] + list(torch.unique(pred_prob_max, sorted=True))
        costs = []
        for threshold in thresholds:
            thresholded_ys = torch.where(pred_prob_max <= threshold, -1 * torch.ones_like(pred_ys), pred_ys)
            costs.append(cost_function(thresholded_ys, ys).item())
        best_idx = np.argmin(costs)
        if costs[best_idx] < best_cost:
            best_cost = costs[best_idx]
            print(f"New best cost: {best_cost} with threshold {thresholds[best_idx]}")
    return best_cost
        

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    # Visualize the optimization history
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()


        

