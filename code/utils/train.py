from Animator import Animator


def train(model, X_train, y_train, X_val, y_val,
          epochs, loss_fn, optimizer,
          stage, draw, log):
    if draw:
        animator = Animator(xlabel='epoch', xlim=[1, epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % stage == 0:
            # 指标
            y_train_pred = (y_pred > 0.5).float()
            y_val_pred = (model(X_val) > 0.5).float()
            train_acc = (y_train_pred == y_train.unsqueeze(1)).float().mean()
            val_acc = (y_val_pred == y_val.unsqueeze(1)).float().mean()
            # 画图，打印
            if draw:
                train_metrics = (loss.item(), train_acc.item())
                animator.add(epoch + 1, train_metrics + (val_acc,))

            if log:
                print("Epoch: {}, Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}"
                      .format(epoch + 1, loss.item(), train_acc.item(), val_acc.item()))
