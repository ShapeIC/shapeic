use std::collections::BTreeMap;

use super::VerificationError;

pub(crate) fn render(
    source: &str,
    values: &BTreeMap<String, String>,
) -> Result<String, VerificationError> {
    let mut output = String::with_capacity(source.len());
    let mut remaining = source;

    while let Some(start) = remaining.find("{{") {
        output.push_str(&remaining[..start]);
        remaining = &remaining[start + 2..];
        let end = remaining.find("}}").ok_or_else(|| {
            VerificationError::Template("unterminated '{{' placeholder".to_owned())
        })?;
        let key = remaining[..end].trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return Err(VerificationError::Template(format!(
                "invalid placeholder '{{{{{key}}}}}'"
            )));
        }
        let value = values
            .get(key)
            .ok_or_else(|| VerificationError::Template(format!("undefined placeholder '{key}'")))?;
        output.push_str(value);
        remaining = &remaining[end + 2..];
    }

    if remaining.contains("}}") {
        return Err(VerificationError::Template(
            "closing '}}' without matching '{{'".to_owned(),
        ));
    }
    output.push_str(remaining);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_known_placeholders_strictly() {
        let values = BTreeMap::from([
            ("width".to_owned(), "5e-6".to_owned()),
            ("model".to_owned(), "nmos".to_owned()),
        ]);
        assert_eq!(
            render("M1 model={{ model }} w={{width}}", &values).expect("render"),
            "M1 model=nmos w=5e-6"
        );
        assert!(matches!(
            render("{{missing}}", &values),
            Err(VerificationError::Template(_))
        ));
        assert!(matches!(
            render("{{width", &values),
            Err(VerificationError::Template(_))
        ));
    }
}
