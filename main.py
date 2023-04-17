from mask_rcnn.model import LitgoModelImpl


if __name__ == "__main__":
    model = LitgoModelImpl(dataset_name="UAVVaste")

    ## Verifty dataset is working
    # print(len(model.dataset))

    # imgs, targets = model.dataset[0]
    # print(imgs.shape)
    # print()
    # for k,v in targets.items():
    #     print(k, v.shape)

    # print(model.dataset.get_num_classes())
    # print(model.dataset.name())
    
    model.train(num_epochs=10, batch_size=2, test_batch_size=1)
