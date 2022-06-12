from ResnetMain import ResnetMainRunner



if __name__ == '__main__':
    model = ResnetMainRunner()
    model.train(n_epochs=2)
    #model.run_inference(validation=True)
    #model.run_inference(validation=False)
