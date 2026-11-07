#ifndef PIPELINE_SHARE_SYNC_H
#define PIPELINE_SHARE_SYNC_H

#include <QByteArray>
#include <QList>
#include <QObject>
#include <QString>
#include <QUrl>

class QNetworkAccessManager;
class QNetworkReply;

// One round of the pipeline sharing protocol (POST <url>/v1/sync) for a title: uploads the records
// the emulator recorded (_PipelineCache/<title>_<version>.kpm, written with
// --record-pipeline-manifest) that neither the manifests built with the emulator, nor the shared
// manifest, nor an earlier round already has, and appends the records the server sends back to
// _PipelineCache/<title>_<version>.shared.kpm, which the emulator loads at start and then reads
// again while the game runs. Files are read and written on a worker thread, the network is
// asynchronous: the interface thread never waits. Failures are reported in Finished() and the
// caller tries again in its next round.
class PipelineShareSync: public QObject {
	Q_OBJECT

public:
	struct Options {
		QUrl    url;              // The server, without /v1/sync.
		QString client_id;        // Random identifier of this installation.
		QString emulator_version; // At most 64 characters are sent.
		QString user_agent;
		QString emulator_dir;     // Holds _PipelineCache and assets/pipelines.
		QString title_id;
		// The version of the game as its param.json says (the emulator names its files after
		// it). With it a round also works before the game ran for the first time: there is no
		// recording to take the version from yet.
		QString game_version;
	};

	explicit PipelineShareSync(Options options, QObject* parent = nullptr);
	~PipelineShareSync() override;

	[[nodiscard]] bool IsRunning() const { return m_running; }
	// Starts a round; nothing happens while one is running. A round that only fetches uploads
	// nothing and needs no recording: it runs before the game starts, so that the shared manifest
	// is there when the emulator loads its manifests.
	void Start(bool fetch_only = false);

signals:
	void Finished(bool ok, const QString& message);

private:
	struct Prepared {
		QString           error;
		bool              nothing_recorded = false;
		QString           version;
		QString           format;
		QString           shared_path;
		QString           assets_path;
		QString           state_path;
		QList<QByteArray> records; // Not uploaded yet, in file order.
		QList<QByteArray> hashes;  // Of `records`.
		quint64           cursor = 0;
		// Reconciliation: the cursor starts at 0 and the server is told what we have.
		bool              reconcile = false;
		QString           assets_digest;
		QList<QByteArray> have; // RecordHash of every record of assets/pipelines and the shared manifest.
	};
	struct Applied {
		QString error;
		int     stored = 0;
	};

	static Prepared Prepare(const Options& options, bool fetch_only);
	static Applied  Apply(const Prepared& prepared, const QList<QByteArray>& received,
	                      const QList<QByteArray>& uploaded, quint64 cursor, bool reconciled);

	void SendNext();
	void OnReply(QNetworkReply* reply);
	void Finish(bool ok, const QString& message);

	Options                m_options;
	QNetworkAccessManager* m_network = nullptr;
	bool                   m_running = false;
	Prepared               m_prepared;
	qsizetype              m_position = 0;
	qsizetype              m_sent_count = 0;
	quint64                m_cursor   = 0;
	bool                   m_more     = false;
	bool                   m_reconciling = false;
	int                    m_requests = 0;
	int                    m_uploaded = 0;
	int                    m_received = 0;
};

#endif // PIPELINE_SHARE_SYNC_H
