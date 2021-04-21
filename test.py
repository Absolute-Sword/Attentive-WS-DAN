from tqdm import tqdm
import time

pbar = tqdm(total=60, unit=' s')
for i in range(60):
    pbar.set_description("seconds have benn lost")  # 进度条前面的描述
    time.sleep(1)
    pbar.update()
    pbar.set_postfix_str("+1s")
