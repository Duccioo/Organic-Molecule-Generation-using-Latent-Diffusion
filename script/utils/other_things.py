def hungarian_algorithm(costs):
    # print(costs)
    # obtain a column vector of minimum row values
    row_mins, _ = torch.min(costs, dim=1, keepdim=True)
    # subtract the tensor of minimum values (broadcasting the minimum value over each row)
    costs = costs - row_mins
    # obtain a row vector of minimum column values
    col_mins, _ = torch.min(costs, dim=0, keepdim=True)
    # subtract the tensor of minimum values (broadcasting the minimum value over each column)
    costs = costs - col_mins
    # proceed with partial assignment
    row_zero_counts = costs.size(1) - torch.count_nonzero(costs, dim=1)
    assigned_columns = []
    assigned_rows = []
    assignment = []
    # assign rows in progressive order of available options
    for opt in range(1, torch.max(row_zero_counts) + 1):
        for i in torch.argwhere(row_zero_counts == opt):
            for j in torch.argwhere(costs[i, :] == 0)[:, 1:]:
                if i.item() not in assigned_rows and j.item() not in assigned_columns:
                    assigned_rows.append(i.item())
                    assigned_columns.append(j.item())
                    assignment.append(torch.concatenate((i, j)))
    # refine assignment until all the rows and columns are assigned
    while len(assignment) < costs.size(0):
        # mark unassigned rows
        marked_rows = list(set(range(costs.size(0))) - set(assigned_rows))
        # build queue of rows to examine
        row_queue = list(set(range(costs.size(0))) - set(assigned_rows))
        # initialize empty list of marked columns
        marked_columns = []
        # initialize empty queue of columns to examine
        column_queue = []
        # examine rows and columns until everything marked is examined
        while len(row_queue) > 0:
            # mark columns with zeros in marked rows
            for j in torch.argwhere(costs[row_queue, :] == 0):
                if j[1].item() not in marked_columns:
                    marked_columns.append(j[1].item())
                    column_queue.append(j[1].item())
            # empty row queue
            row_queue = []
            # mark assigned rows with assignment on marked columns in the queue
            for t in assignment:
                if t[1].item() in column_queue and t[0].item() not in marked_rows:
                    marked_rows.append(t[0].item())
                    row_queue.append(t[0].item())
            # empty column queue
            column_queue = []
        # obtain minimum uncovered element (on marked rows and unmarked columns)

        try:

            min_value = torch.min(
                costs[marked_rows, :][:, list(set(range(costs.size(1))) - set(marked_columns))]
            )
        except:
            min_value = 0

        # subtract minimum value from uncovered elements
        # print(min_value.item())
        # print(costs[0, 0])

        for i in marked_rows:
            for j in list(set(range(costs.size(1))) - set(marked_columns)):
                costs[i, j] = costs[i, j] - min_value
        # and minimum value to double-covered elements
        for i in list(set(range(costs.size(0))) - set(marked_rows)):
            for j in marked_columns:
                costs[i, j] = costs[i, j] + min_value
        # re-assign everything
        row_zero_counts = costs.size(1) - torch.count_nonzero(costs, dim=1)
        assigned_columns = []
        assigned_rows = []
        assignment = []
        # assign rows in progressive order of available options
        for opt in range(1, torch.max(row_zero_counts) + 1):
            for i in torch.argwhere(row_zero_counts == opt):
                for j in torch.argwhere(costs[i, :] == 0)[:, 1:]:
                    if i.item() not in assigned_rows and j.item() not in assigned_columns:
                        assigned_rows.append(i.item())
                        assigned_columns.append(j.item())
                        assignment.append(torch.concatenate((i, j)))
    # obtain final assignment tensor
    assignment = [torch.reshape(a, (1, 2)) for a in assignment]
    assignment = torch.concatenate(assignment, dim=0)
    # return row indices and column inices as separate tensors
    # print("MA QUI CI SONO?")
    return assignment[:, 0], assignment[:, 1]
