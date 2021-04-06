import torch




def train_model_contrastive(model,
                            loader,
                            optimizer,
                            device,
                            criterion=F.nll_loss,
                            complexity_loss_fn=None,
                            batches_in_epoch=sys.maxsize,
                            active_classes=None,
                            pre_batch_callback=None,
                            post_batch_callback=None,
                            transform_to_device_fn=None,
                            progress_bar=None,):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: train dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param active_classes: a list of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
    :type active_classes: list of int or None
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None

    :return: mean loss for epoch
    :rtype: float
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    # Check if training with Apex Mixed Precision
    # FIXME: There should be another way to check if 'amp' is enabled
    use_amp = hasattr(optimizer, "_amp_stash")
    try:
        from apex import amp
    except ImportError:
        if use_amp:
            raise ImportError(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex")


    t0 = time.time()
    for batch_idx, (model_input, label) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        if transform_to_device_fn is None:
            model_input = model_input.to(device, non_blocking=async_gpu)
            label = label.to(device, non_blocking=async_gpu)
        else:
            model_input, label = transform_to_device_fn(model_input, label, device,
                                                  non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        module_loss_vals, _, _, accuracy = model(model_input, label)
        module_loss_vals = torch.mean(module_loss_vals, 0)  # take mean over outputs of different GPUs
        error_loss = torch.sum(module_loss_vals) #sum losses of all modules
        accuracy = torch.mean(accuracy, 0)

        # if cur_train_module != opt.model_splits and opt.model_splits > 1:
        #     loss = loss[cur_train_module].unsqueeze(0)

        # loop through the losses of the modules and do gradient descent
        # total_loss.backward()

        optimizer.zero_grad()
        error_loss.backward()

        del model_input, label

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(error_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            error_loss.backward()

        t3 = time.time()

        # Compute and backpropagate the complexity loss. This happens after
        # error loss has backpropagated, freeing its computation graph, so the
        # two loss functions don't compete for memory.
        complexity_loss = (complexity_loss_fn(model)
                           if complexity_loss_fn is not None
                           else None)
        if complexity_loss is not None:
            if use_amp:
                with amp.scale_loss(complexity_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                complexity_loss.backward()

        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           "complexity loss forward/backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3, t5 - t4)
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                complexity_loss=(complexity_loss.detach()
                                                 if complexity_loss is not None
                                                 else None),
                                batch_idx=batch_idx,
                                num_images=num_images,
                                time_string=time_string)
        del error_loss, complexity_loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()
