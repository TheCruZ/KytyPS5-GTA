#include "pipelineShareSync.h"

#include <QByteArrayView>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>
#include <QSet>
#include <QtConcurrent>

#include <algorithm>

namespace {

constexpr char       FORMAT_PREFIX[]  = "# kyty pipeline manifest v";
constexpr qsizetype  MAX_RECORD_BYTES = 8192;
constexpr qsizetype  MAX_REQUEST_RECORDS = 5000;
constexpr qsizetype  MAX_REQUEST_BYTES   = 2 * 1024 * 1024;
constexpr qint64     MAX_RESPONSE_BYTES  = 64 * 1024 * 1024;
constexpr qint64     MAX_FILE_BYTES      = 512ll * 1024 * 1024;
constexpr int        MAX_REQUESTS_PER_ROUND = 64;
constexpr int        REQUEST_TIMEOUT_MS  = 30000;
constexpr int        PROTOCOL            = 2;
// Hashes of the records the client has that one request carries (the server takes up to
// 200000 per partition and the body is limited to 4 MiB).
constexpr qsizetype  MAX_HAVE            = 150000;

QString LogPrefix() {
	return QStringLiteral("Pipeline sharing: ");
}

// A record the manifest reader would take: one line of printable ASCII that starts with the
// letter of a record type.
bool ValidRecord(const QByteArray& line) {
	if (line.size() < 3 || line.size() > MAX_RECORD_BYTES || line[1] != ' ' ||
	    (line[0] != 'M' && line[0] != 'P' && line[0] != 'T')) {
		return false;
	}
	return std::none_of(line.begin(), line.end(),
	                    [](char c) { return static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) > 0x7e; });
}

QByteArray RecordHash(const QByteArray& line) {
	return QCryptographicHash::hash(line, QCryptographicHash::Sha256).left(8).toHex();
}

// The complete lines of a manifest file: the first line (the format) in `header` and the records
// in file order. A last line without its newline is still being written and is left out.
QList<QByteArray> ReadLines(const QString& path, QByteArray* header = nullptr) {
	QList<QByteArray> records;
	QFile             file(path);
	if (!file.open(QIODevice::ReadOnly) || file.size() > MAX_FILE_BYTES) {
		return records;
	}
	const QByteArray data = file.readAll();
	qsizetype        start = 0;
	bool             first = true;
	while (start < data.size()) {
		const auto end = data.indexOf('\n', start);
		if (end < 0) {
			break;
		}
		auto line = data.mid(start, end - start);
		start     = end + 1;
		if (line.endsWith('\r')) {
			line.chop(1);
		}
		if (line.isEmpty()) {
			continue;
		}
		if (first && header != nullptr) {
			*header = line;
		}
		first = false;
		if (line[0] != '#') {
			records.append(line);
		}
	}
	return records;
}

struct State {
	quint64          cursor = 0;
	QSet<QByteArray> uploaded;
	// SHA-256 of the assets/pipelines file the cursor refers to ("" when there was none). A
	// state without it is from a client that did not reconcile.
	bool             has_assets = false;
	QString          assets;
};

// SHA-256 of a manifest file, to notice that the emulator was updated (another release brings
// other manifests); "" when the file does not exist.
QString FileDigest(const QString& path) {
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly) || file.size() > MAX_FILE_BYTES) {
		return {};
	}
	QCryptographicHash hash(QCryptographicHash::Sha256);
	hash.addData(&file);
	return QString::fromLatin1(hash.result().toHex());
}

State LoadState(const QString& path) {
	State state;
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly)) {
		return state;
	}
	const auto document = QJsonDocument::fromJson(file.readAll());
	if (!document.isObject()) {
		return state;
	}
	const auto object = document.object();
	state.cursor      = object.value(QStringLiteral("cursor")).toString().toULongLong();
	state.has_assets  = object.contains(QStringLiteral("assets"));
	state.assets      = object.value(QStringLiteral("assets")).toString();
	for (const auto& value: object.value(QStringLiteral("uploaded")).toArray()) {
		state.uploaded.insert(value.toString().toLatin1());
	}
	return state;
}

bool SaveState(const QString& path, const State& state) {
	QJsonArray uploaded;
	for (const auto& hash: state.uploaded) {
		uploaded.append(QString::fromLatin1(hash));
	}
	QJsonObject object;
	object.insert(QStringLiteral("cursor"), QString::number(state.cursor));
	if (state.has_assets) {
		object.insert(QStringLiteral("assets"), state.assets);
	}
	object.insert(QStringLiteral("uploaded"), uploaded);
	QDir().mkpath(QFileInfo(path).absolutePath());
	QSaveFile file(path);
	if (!file.open(QIODevice::WriteOnly)) {
		return false;
	}
	file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
	return file.commit();
}

// The version of the game in a file name, written as the emulator does (anything but letters,
// digits, '-', '_' and '.' becomes '_'); empty when it is none or the server would refuse it.
QString FileNameVersion(QString version) {
	version = version.trimmed();
	for (auto& c: version) {
		if (!(c.isLetterOrNumber() && c.unicode() < 0x80) && c != QLatin1Char('-') &&
		    c != QLatin1Char('_') && c != QLatin1Char('.')) {
			c = QLatin1Char('_');
		}
	}
	return version.size() <= 32 ? version : QString();
}

// The format line of the manifests of this emulator: that of the first of these files that has
// one, then of any manifest built with the emulator; the first format of all when there is none.
QString ManifestFormat(const QString& emulator_dir, const QString& first, const QString& second,
                       const QString& third) {
	const auto format_of = [](const QString& path) {
		QFile file(path);
		if (path.isEmpty() || !file.open(QIODevice::ReadOnly)) {
			return QString();
		}
		auto line = file.readLine(256).trimmed();
		return line.startsWith(FORMAT_PREFIX) ? QString::fromLatin1(line) : QString();
	};
	for (const auto& path: {first, second, third}) {
		if (auto format = format_of(path); !format.isEmpty()) {
			return format;
		}
	}
	const QDir assets(QDir(emulator_dir).filePath(QStringLiteral("assets/pipelines")));
	for (const auto& entry: assets.entryInfoList({QStringLiteral("*.kpm")}, QDir::Files)) {
		if (auto format = format_of(entry.filePath()); !format.isEmpty()) {
			return format;
		}
	}
	return QString::fromLatin1(FORMAT_PREFIX) + QLatin1Char('1');
}

} // namespace

PipelineShareSync::PipelineShareSync(Options options, QObject* parent)
    : QObject(parent), m_options(std::move(options)), m_network(new QNetworkAccessManager(this)) {}

PipelineShareSync::~PipelineShareSync() = default;

PipelineShareSync::Prepared PipelineShareSync::Prepare(const Options& options, bool fetch_only) {
	Prepared    prepared;
	const QDir  cache(QDir(options.emulator_dir).filePath(QStringLiteral("_PipelineCache")));
	const auto  prefix = options.title_id + QLatin1Char('_');
	// The emulator names its files <title>_<version> from the game's version; the launcher knows
	// that version before the game has ever run.
	const auto  game_version = FileNameVersion(options.game_version);
	QFileInfo   recorded;
	QString     stem;
	if (!game_version.isEmpty()) {
		stem     = prefix + game_version;
		recorded = QFileInfo(cache.filePath(stem + QStringLiteral(".kpm")));
		prepared.version = game_version;
	} else {
		// The recording of the newest version: <title>_<version>.kpm, not the shared manifest.
		for (const auto& entry: cache.entryInfoList({prefix + QStringLiteral("*.kpm")}, QDir::Files)) {
			if (entry.fileName().endsWith(QStringLiteral(".shared.kpm"))) {
				continue;
			}
			if (!recorded.exists() || entry.lastModified() > recorded.lastModified()) {
				recorded = entry;
			}
		}
		if (!recorded.exists()) {
			prepared.nothing_recorded = true;
			return prepared;
		}
		const auto name  = recorded.fileName();
		prepared.version = name.mid(prefix.size(), name.size() - prefix.size() - 4);
		if (prepared.version.isEmpty()) {
			prepared.nothing_recorded = true;
			return prepared;
		}
		stem = name.left(name.size() - 4);
	}
	const auto name      = stem + QStringLiteral(".kpm");
	prepared.shared_path = cache.filePath(stem + QStringLiteral(".shared.kpm"));
	prepared.assets_path =
	    QDir(options.emulator_dir).filePath(QStringLiteral("assets/pipelines/") + name);
	prepared.state_path = cache.filePath(QStringLiteral("share/") + stem + QStringLiteral(".json"));

	QList<QByteArray> lines;
	if (recorded.exists() && !fetch_only) {
		QByteArray header;
		lines = ReadLines(recorded.filePath(), &header);
		if (!header.startsWith(FORMAT_PREFIX)) {
			prepared.error = QStringLiteral("%1 is not a pipeline manifest").arg(name);
			return prepared;
		}
		prepared.format = QString::fromLatin1(header);
	} else {
		// Nothing to send (a round before the game, or one the game has not written a recording
		// for yet): the format comes from the other manifests, and the round brings what the
		// others found.
		prepared.format = ManifestFormat(options.emulator_dir, recorded.filePath(),
		                                 prepared.assets_path, prepared.shared_path);
	}

	QSet<QByteArray> known;
	for (const auto& line: ReadLines(prepared.assets_path)) {
		known.insert(line);
	}
	for (const auto& line: ReadLines(prepared.shared_path)) {
		known.insert(line);
	}
	const auto state = LoadState(prepared.state_path);
	// With other manifests than the ones the cursor was taken with (a new installation, an
	// update of the emulator, a client that never did this), start over and tell the server what
	// we have, so that it only sends the rest.
	prepared.assets_digest = FileDigest(prepared.assets_path);
	prepared.reconcile     = state.cursor == 0 || !state.has_assets || state.assets != prepared.assets_digest;
	prepared.cursor        = prepared.reconcile ? 0 : state.cursor;
	if (prepared.reconcile) {
		for (const auto& line: known) {
			if (prepared.have.size() >= MAX_HAVE) {
				break;
			}
			prepared.have.append(RecordHash(line));
		}
	}
	for (const auto& line: lines) {
		if (known.contains(line) || !ValidRecord(line)) {
			continue;
		}
		const auto hash = RecordHash(line);
		if (!state.uploaded.contains(hash)) {
			prepared.records.append(line);
			prepared.hashes.append(hash);
		}
	}
	return prepared;
}

PipelineShareSync::Applied PipelineShareSync::Apply(const Prepared&         prepared,
                                                    const QList<QByteArray>& received,
                                                    const QList<QByteArray>& uploaded,
                                                    quint64                  cursor,
                                                    bool                     reconciled) {
	Applied applied;
	// The shared manifest keeps the format of the recording; one of another format is left alone.
	QByteArray header;
	auto       known_lines = ReadLines(prepared.shared_path, &header);
	const bool exists      = QFileInfo::exists(prepared.shared_path);
	if (exists && !header.isEmpty() && header != prepared.format.toLatin1()) {
		applied.error = QStringLiteral("the shared manifest has another format");
		return applied;
	}
	QSet<QByteArray> known(known_lines.begin(), known_lines.end());
	for (const auto& line: ReadLines(prepared.assets_path)) {
		known.insert(line);
	}
	QByteArray added;
	for (const auto& line: received) {
		if (ValidRecord(line) && !known.contains(line)) {
			known.insert(line);
			added += line;
			added += '\n';
			applied.stored++;
		}
	}
	if (!added.isEmpty()) {
		QDir().mkpath(QFileInfo(prepared.shared_path).absolutePath());
		QFile file(prepared.shared_path);
		if (!file.open(QIODevice::Append)) {
			applied.error = QStringLiteral("cannot write %1").arg(prepared.shared_path);
			return applied;
		}
		QByteArray out;
		if (file.size() == 0) {
			out = prepared.format.toLatin1() + '\n';
		} else {
			// A write that was cut short leaves a line without its end.
			QFile check(prepared.shared_path);
			if (check.open(QIODevice::ReadOnly) && check.seek(check.size() - 1) &&
			    check.read(1) != "\n") {
				out = "\n";
			}
		}
		out += added;
		if (file.write(out) != out.size()) {
			applied.error = QStringLiteral("cannot write %1").arg(prepared.shared_path);
			return applied;
		}
	}
	auto state   = LoadState(prepared.state_path);
	state.cursor = cursor;
	if (reconciled) {
		state.has_assets = true;
		state.assets     = prepared.assets_digest;
	}
	for (const auto& hash: uploaded) {
		state.uploaded.insert(hash);
	}
	if (!SaveState(prepared.state_path, state)) {
		applied.error = QStringLiteral("cannot write %1").arg(prepared.state_path);
	}
	return applied;
}

void PipelineShareSync::Start(bool fetch_only) {
	if (m_running) {
		return;
	}
	m_running  = true;
	m_requests = m_uploaded = m_received = 0;
	m_position                         = 0;
	m_more                             = false;
	auto* watcher = new QFutureWatcher<Prepared>(this);
	connect(watcher, &QFutureWatcher<Prepared>::finished, this, [this, watcher] {
		m_prepared = watcher->result();
		watcher->deleteLater();
		if (!m_prepared.error.isEmpty()) {
			Finish(false, m_prepared.error);
		} else if (m_prepared.nothing_recorded) {
			Finish(true, QStringLiteral("nothing recorded yet"));
		} else {
			m_cursor      = m_prepared.cursor;
			m_reconciling = m_prepared.reconcile;
			SendNext();
		}
	});
	watcher->setFuture(QtConcurrent::run(&PipelineShareSync::Prepare, m_options, fetch_only));
}

void PipelineShareSync::SendNext() {
	if (m_requests++ >= MAX_REQUESTS_PER_ROUND) {
		Finish(true, QStringLiteral("stopped after %1 requests").arg(MAX_REQUESTS_PER_ROUND));
		return;
	}
	// The records of this request: none while the server has more to send.
	QJsonArray records;
	qsizetype  bytes = 0;
	m_sent_count     = 0;
	while (!m_more && m_position + m_sent_count < m_prepared.records.size() &&
	       m_sent_count < MAX_REQUEST_RECORDS && bytes < MAX_REQUEST_BYTES) {
		const auto& record = m_prepared.records[m_position + m_sent_count];
		records.append(QString::fromLatin1(record));
		bytes += record.size();
		m_sent_count++;
	}
	QJsonObject body;
	body.insert(QStringLiteral("client"), m_options.client_id);
	body.insert(QStringLiteral("emulator"), m_options.emulator_version.left(64));
	body.insert(QStringLiteral("format"), m_prepared.format);
	body.insert(QStringLiteral("title"), m_options.title_id);
	body.insert(QStringLiteral("version"), m_prepared.version);
	body.insert(QStringLiteral("cursor"), static_cast<qint64>(m_cursor));
	body.insert(QStringLiteral("records"), records);
	body.insert(QStringLiteral("proto"), PROTOCOL);
	if (m_reconciling) {
		// The reply leaves out these; it is repeated in every page until `more` is false.
		QJsonArray have;
		for (const auto& hash: m_prepared.have) {
			have.append(QString::fromLatin1(hash));
		}
		body.insert(QStringLiteral("have"), have);
	}

	auto url = m_options.url;
	url.setPath(url.path() + QStringLiteral("/v1/sync"));
	QNetworkRequest request(url);
	request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
	request.setRawHeader("User-Agent", m_options.user_agent.toUtf8());
	request.setRawHeader("Accept", "application/json");
	request.setTransferTimeout(REQUEST_TIMEOUT_MS);
	// A redirect would turn the POST into a GET: the address is the server's.
	request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
	                     QNetworkRequest::ManualRedirectPolicy);
	auto* reply = m_network->post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
	connect(reply, &QNetworkReply::finished, this, [this, reply] { OnReply(reply); });
}

void PipelineShareSync::OnReply(QNetworkReply* reply) {
	reply->deleteLater();
	const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
	if (reply->error() != QNetworkReply::NoError || status != 200) {
		Finish(false, QStringLiteral("the server answered %1 (%2)")
		                  .arg(status)
		                  .arg(reply->errorString()));
		return;
	}
	if (reply->bytesAvailable() > MAX_RESPONSE_BYTES) {
		Finish(false, QStringLiteral("the answer is too large"));
		return;
	}
	QJsonParseError error;
	const auto      document = QJsonDocument::fromJson(reply->readAll(), &error);
	if (error.error != QJsonParseError::NoError || !document.isObject()) {
		// A page of a proxy (a challenge, an error) instead of the server's answer.
		Finish(false, QStringLiteral("the answer is not JSON"));
		return;
	}
	const auto object = document.object();
	if (!object.value(QStringLiteral("cursor")).isDouble() ||
	    !object.value(QStringLiteral("records")).isArray()) {
		Finish(false, QStringLiteral("the answer is not a sync answer"));
		return;
	}
	const auto cursor = static_cast<quint64>(object.value(QStringLiteral("cursor")).toDouble());
	QList<QByteArray> received;
	for (const auto& value: object.value(QStringLiteral("records")).toArray()) {
		if (value.isString()) {
			received.append(value.toString().toLatin1());
		}
	}
	m_more = object.value(QStringLiteral("more")).toBool();
	QList<QByteArray> uploaded = m_prepared.hashes.mid(m_position, m_sent_count);
	m_position += m_sent_count;
	m_uploaded += static_cast<int>(m_sent_count);
	m_cursor = cursor;
	// The reconciliation ends with the last page; the new manifests are noted then.
	const bool reconciled = m_reconciling && !m_more;
	if (reconciled) {
		m_reconciling = false;
	}

	auto* watcher = new QFutureWatcher<Applied>(this);
	connect(watcher, &QFutureWatcher<Applied>::finished, this, [this, watcher] {
		const auto applied = watcher->result();
		watcher->deleteLater();
		m_received += applied.stored;
		if (!applied.error.isEmpty()) {
			Finish(false, applied.error);
		} else if (m_more || m_position < m_prepared.records.size()) {
			SendNext();
		} else {
			Finish(true, QStringLiteral("%1 records uploaded, %2 received")
			                 .arg(m_uploaded)
			                 .arg(m_received));
		}
	});
	watcher->setFuture(QtConcurrent::run(&PipelineShareSync::Apply, m_prepared, received, uploaded,
	                                     cursor, reconciled));
}

void PipelineShareSync::Finish(bool ok, const QString& message) {
	m_running  = false;
	m_prepared = {};
	emit Finished(ok, LogPrefix() + m_options.title_id + QStringLiteral(": ") + message);
}
