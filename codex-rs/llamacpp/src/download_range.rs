//! Validate HTTP range metadata before appending to an existing download.
use std::io;

pub(crate) fn resumed_total(
    status: u16,
    header: Option<&str>,
    offset: u64,
) -> io::Result<Option<u64>> {
    let invalid = || {
        io::Error::other("Download returned inconsistent Content-Range; partial download preserved")
    };
    if status == 416 {
        let total = header
            .and_then(|v| v.strip_prefix("bytes */"))
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(invalid)?;
        return if offset > 0 && offset == total {
            Ok(Some(total))
        } else {
            Err(invalid())
        };
    }
    if status != 206 {
        return Ok(None);
    }
    let range = header
        .and_then(|v| v.strip_prefix("bytes "))
        .ok_or_else(invalid)?;
    let (bounds, total) = range.split_once('/').ok_or_else(invalid)?;
    let (start, end) = bounds.split_once('-').ok_or_else(invalid)?;
    let start = start.parse::<u64>().map_err(|_| invalid())?;
    let end = end.parse::<u64>().map_err(|_| invalid())?;
    let total = total.parse::<u64>().map_err(|_| invalid())?;
    if start != offset || end < start || end >= total {
        return Err(invalid());
    }
    Ok(Some(total))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_exact_complete_parts_accept_416() {
        assert_eq!(resumed_total(416, Some("bytes */8"), 8).unwrap(), Some(8));
        for (header, offset) in [
            (Some("bytes */8"), 9),
            (Some("bytes */8"), 4),
            (None, 8),
            (Some("bytes */0"), 0),
        ] {
            assert!(resumed_total(416, header, offset).is_err());
        }
    }
    #[test]
    fn ranges_must_start_at_the_resume_offset() {
        assert_eq!(resumed_total(206, Some("bytes 4-7/8"), 4).unwrap(), Some(8));
        for value in [
            "bytes 0-7/8",
            "bytes 5-7/8",
            "bytes 4-8/8",
            "bytes 4-7/*",
            "garbage",
        ] {
            assert!(resumed_total(206, Some(value), 4).is_err());
        }
    }
}

#[cfg(test)]
mod network_tests {
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::header;
    use wiremock::matchers::method;
    #[tokio::test]
    async fn invalid_ranges_preserve_partial_files_without_a_checksum() {
        for (index, status, range) in [(0, 416, "bytes */3"), (1, 206, "bytes 0-3/8"), (2, 204, "")]
        {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(header("range", "bytes=4-"))
                .respond_with(ResponseTemplate::new(status).insert_header("content-range", range))
                .expect(1)
                .mount(&server)
                .await;
            let directory =
                std::env::temp_dir().join(format!("millie-range-{}-{index}", std::process::id()));
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join("weights.gguf");
            let partial = directory.join("weights.gguf.part");
            std::fs::write(&partial, b"part").unwrap();
            assert!(
                crate::ensure_local_file(&path, Some(&server.uri()), None, "test")
                    .await
                    .is_err()
            );
            assert!(!path.exists());
            assert_eq!(std::fs::read(&partial).unwrap(), b"part");
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}
