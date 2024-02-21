import pandas as pd

excel_path = r"E:\OneDrive\毕业论文\数据\原始_测速数据_E3.xlsx"

out_path = r".\out\output.csv"
# 读取Excel文件中的所有工作表
excel_data = pd.read_excel(excel_path, sheet_name=None)


out_columns = ['序号', '坡面名称', '冲刷阶段', '时间', '流速上', '测速时间①', '测速位置①', '流速中', '测速时间②', '测速位置②', '流速下', '测速时间③', '测速位置③']
# 创建一个空的DataFrame来存储处理后的数据
processed_data = pd.DataFrame(columns=out_columns)

sheet_names = ["第一段", "第二段", "第三段", "第四段"]

new_index = -1
# 遍历每个工作表
for sheet_name, df in excel_data.items():
    # 在这里进行你的数据处理操作，可以使用pandas提供的各种函数和方法
    # 例如，打印工作表的前5行数据
    print(f"正在处理工作表：{sheet_name}")

    if sheet_name not in sheet_names:
        continue

    # df['平均测速时刻'] = df['平均测速时刻'].astype(str)

    last_round = -1
    round_times = 0

    # 遍历每一行数据
    for index, row in df.iterrows():
        cur_round = row['测速轮次']

        # slope_name = row['坡面名称']

        if cur_round == last_round:
            round_times += 1
        else:
            round_times = 0
            new_index += 1

        velocity = row['流速值']
        measure_time = str(row['平均测速时刻'])
        measure_pos = row['平均测速位置']

        new_measure_data = {}

        if round_times == 0:
            # measure_time = measure_time.astype(str)

            # todo 根据上坡段的测速时间判断，本轮应该是属于第几分钟的测速
            # 第几分钟的测速
            time = int(measure_time.split(':')[0]) + 1
            print(time)

            new_measure_data = {
                '序号': cur_round,
                '冲刷阶段': sheet_name,
                '时间': time,
                '流速上': velocity,
                '测速时间①': measure_time,
                '测速位置①': measure_pos
            }
        elif round_times == 1:
            new_measure_data = {
                '序号': cur_round,
                '流速中': velocity,
                '测速时间②': measure_time,
                '测速位置②': measure_pos
            }
        elif round_times == 2:
            new_measure_data = {
                '序号': cur_round,
                '流速下': velocity,
                '测速时间③': measure_time,
                '测速位置③': measure_pos
            }

        # 使用loc在指定的行索引插入数据

        # insert_columns = ['序号', '流速上', '测速时间①', '测速位置①']

        # processed_data.loc[new_index, insert_columns] = new_measure_data.values()

        # 使用at方法在指定的行索引和列标签插入数据
        for col in new_measure_data:
            processed_data.at[new_index, col] = new_measure_data[col]

        last_round = cur_round

        # todo 根据测速轮次进行上中下坡段的流速的划分，同一轮次的在结果数据的同一行
        # print(f"序号：{row['序号']}")
        # print(f"序号：{row['序号']}")
        # print(f"坡面名称：{row['坡面名称']}")

    # break


print(processed_data)


# 将处理后的数据导出为CSV文件
processed_data.to_csv(out_path, encoding='utf-8-sig', index=False)
print(f'输出结果到：{out_path}')