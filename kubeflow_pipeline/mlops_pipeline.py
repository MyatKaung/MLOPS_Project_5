from kfp import dsl

@dsl.container_component
def data_processing_op():
    return dsl.ContainerSpec(
        image="myatkaung/mlops-project5:latest",
        command=["python", "src/data_processing.py"],
    )

@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image="myatkaung/mlops-project5:latest",
        command=["python", "src/model_training.py"],
    )
@dsl.pipeline(
    name="MLops pipeline",
    description="MLops pipeline for healthcare survival prediction"
)
def mlops_pipeline():
    # Create data processing step
    data_proc = data_processing_op()

    # Create model training step
    model_train = model_training_op()
    # Set dependencies
    model_train.after(data_proc)
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path='mlops_pipeline.yaml'
    )
