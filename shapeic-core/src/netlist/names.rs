pub fn small_signal_param_name(param: &str, instance: &str, branch: &str) -> String {
    format!(
        "{}__{}__{}",
        sanitize_name(param),
        sanitize_name(instance),
        sanitize_name(branch)
    )
}

/// Scopes one compact-model parameter to its macro instance path.
pub fn compact_model_param_name(param: &str, instance: &str) -> String {
    format!("{}__{}", sanitize_name(param), sanitize_name(instance))
}
pub fn small_signal_element_name(
    spice_prefix: &str,
    param: &str,
    instance: &str,
    branch: &str,
) -> String {
    format!(
        "{}_{}",
        sanitize_name(spice_prefix),
        small_signal_param_name(param, instance, branch)
    )
}
fn sanitize_name(value: &str) -> String {
    value.trim().replace('-', "_")
}
