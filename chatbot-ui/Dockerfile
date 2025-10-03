# Stage 1: Use the official AWS Lambda Python 3.13 image as a base
FROM public.ecr.aws/lambda/python:3.13

# Copy the requirements file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install the Python dependencies from requirements.txt
# Using --target ensures they are installed in the right directory
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy the rest of your Lambda function code into the container
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the command to run when the container starts.
# This tells Lambda where your handler function is.
CMD [ "lambda_function.lambda_handler" ]