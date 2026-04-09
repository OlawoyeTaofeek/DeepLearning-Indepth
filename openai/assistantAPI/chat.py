from datetime import datetime
import time
import logging
from main import client
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


thread_id = "thread_YkWyz1OJThXlaNQrsYPTtPmf"
assistant_id = "asst_MiyPqmB7UGsO3TVgJynIy45T"

message = "Whats the best way to get in shape for summer if I am currently heavy with a big belly?"

messages = client.beta.threads.messages.create(
    thread_id=thread_id,
    role="user",
    content=message
)

## Now Run the assistant in the thread
runs = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant_id,
    instructions="You are a personal trainer assistant. You help users create workout plans, provide exercise recommendations, and answer fitness-related questions. You have trained high-caliber athletes and movie stars. Always be supportive and encouraging in your responses.",
)

def wait_for_run_completion(client, thread_id, run_id, sleep_interval=1):
    """
    
    waits for a run to complete and returns the final run object
    :param thread_id: the id of the thread the run is in
    :param run_id: the id of the run to wait for
    :param sleep_interval: Time in seconds to wait between checks
    """
    while True:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                elapsed_time = run.completed_at - run.created_at
                print(f"Run completed in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                logging.info(f"Run {run_id} completed in {elapsed_time} seconds")

                # Get the messages once Run is completed
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                # for msg in messages.data:
                #     print(f"{msg.role}: {msg.content}")
                response = messages.data[0].content[0].text.value
                print(F"Assistant Response: {response}")
                break
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"n error occured while retriving the run: {e}")
            break
        logging.info("Waiting for run to complete.....")
        time.sleep(sleep_interval)      


## ====== RUN ======
wait_for_run_completion(client=client, thread_id=thread_id, run_id=runs.id)

# === Steps ====
run_steps = client.beta.threads.runs.steps.list(
    thread_id=thread_id,
    run_id=runs.id
)

for step in run_steps.data:

    print(f"Step ID: {step.id}")
    print(f"Step Type: {step.type}")
    print(f"Step Status: {step.status}")
    print(f"Created At: {step.created_at}")
    print(f"Completed At: {step.completed_at}")

    print("Step Details:")

    print(step.step_details)

    print("------")