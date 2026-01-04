def compute_rollout_pass_rate(
    refined_count,
    hinted_batch,
    zero_pass_count,
    all_pass_count,
    some_pass_count,
    nopass_refined_count,
    some_pass_refined_count,
    all_pass_refined_count,
    og_response_lens,
    new_response_lens,
    hinted_uids_len,
):
    rollout_metrics = {}
    # refined count
    rollout_metrics["rollout/refined_count"] = refined_count
    rollout_metrics["rollout/refined_ratio"] = refined_count / len(hinted_batch) if len(hinted_batch) > 0 else 0
    rollout_metrics["rollout/zero_pass_count"] = zero_pass_count
    rollout_metrics["rollout/all_pass_count"] = all_pass_count
    rollout_metrics["rollout/some_pass_count"] = some_pass_count

    rollout_metrics["rollout/nopass_refined_count"] = nopass_refined_count
    rollout_metrics["rollout/nopass_refined_ratio"] = nopass_refined_count / zero_pass_count if zero_pass_count > 0 else 0
    rollout_metrics["rollout/some_pass_refined_count"] = some_pass_refined_count
    rollout_metrics["rollout/some_pass_refined_ratio"] = some_pass_refined_count / some_pass_count if some_pass_count > 0 else 0
    rollout_metrics["rollout/all_pass_refined_count"] = all_pass_refined_count
    rollout_metrics["rollout/all_pass_refined_ratio"] = all_pass_refined_count / all_pass_count if all_pass_count > 0 else 0
    
    # response length
    if len(og_response_lens) > 0:
        avg_og_len = sum(og_response_lens) / len(og_response_lens)
        avg_new_len = sum(new_response_lens) / len(new_response_lens)
        rollout_metrics["rollout/og_response_lens"] = avg_og_len
        rollout_metrics["rollout/new_response_lens"] = avg_new_len
        rollout_metrics["rollout/response_len_ratio"] = avg_new_len / avg_og_len if avg_og_len > 0 else 0

    print(f"[All] Refined {refined_count} samples out of {hinted_uids_len}")
    print(f"[All] Avg og response len: {avg_og_len}, avg new response len: {avg_new_len}, response len ratio: {avg_new_len / avg_og_len if avg_og_len > 0 else 0}")

    print(f"[NoPass] Refined {nopass_refined_count} samples out of {zero_pass_count}")
    print(f"[SomePass] Refined {some_pass_refined_count} samples out of {some_pass_count}")
    print(f"[AllPass] Refined {all_pass_refined_count} samples out of {all_pass_count}")

    return rollout_metrics